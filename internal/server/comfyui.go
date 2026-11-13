package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/hint"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// ComfyUI compatibility endpoint. ComfyUI is a web app rather than an
// OpenAI-compatible server, so it is served through its own passthrough bound
// to the reserved config.ComfyUIModelID model. Routes are registered in
// server.go; the model's compatibility settings are applied while loading
// config.

func handleComfyUIRedirect(w http.ResponseWriter, r *http.Request) {
	// Relative Location so the redirect keeps a reverse proxy's path prefix
	// (see redirectRelative).
	location := "comfyui/"
	if r.URL.RawQuery != "" {
		location += "?" + r.URL.RawQuery
	}
	status := http.StatusPermanentRedirect
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		status = http.StatusMovedPermanently
	}
	redirectRelative(w, location, status)
}

// comfyUIIgnorePaths are the paths under /comfyui/ that may not start an
// unloaded model. An open ComfyUI tab retries them on its own, so letting them
// swap the model in would undo every unload while a browser is pointed at it.
//
// This is the built-in equivalent of the upstream.ignorePaths a second ComfyUI
// instance needs when it is served through /upstream/<model>/ instead, so the
// two lists are kept the same: the shared static-asset default, the frontend
// websocket, and its job polling.
var comfyUIIgnorePaths = append(
	config.DefaultUpstreamIgnorePaths(),
	regexp.MustCompile(`^/ws(/|$)`),
	regexp.MustCompile(`^/api/jobs$`),
)

// comfyUIIgnoresPath reports whether path is one that may not start the model.
// path is the decoded path only, so query parameters such as ?clientId= never
// affect the match.
func comfyUIIgnoresPath(path string) bool {
	for _, re := range comfyUIIgnorePaths {
		if re.MatchString(path) {
			return true
		}
	}
	return false
}

// handleComfyUI proxies requests under /comfyui/ to the fixed local
// ComfyUI model. Its compatibility settings are applied while loading config.
func (s *Server) handleComfyUI(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.cfg.Models[config.ComfyUIModelID]; !ok || !s.local.Handles(config.ComfyUIModelID) {
		swaputil.SendResponse(w, r, http.StatusNotFound, "local model "+config.ComfyUIModelID+" not found")
		return
	}

	// Strip the /comfyui prefix before forwarding. URL.Path and PathValue are
	// decoded, so retain the matching escaped suffix in RawPath exactly as the
	// generic /upstream handler does.
	remainingPath := "/" + strings.TrimPrefix(r.PathValue("comfyPath"), "/")
	escapedRemaining := swaputil.EscapedPathSuffix(r.URL.EscapedPath(), "/comfyui")
	r.URL.Path = remainingPath
	r.URL.RawPath = escapedRemaining

	// Only a GET is ignored. A write is a deliberate action even on an ignored
	// path, so queueing a job still starts the model.
	if r.Method == http.MethodGet && comfyUIIgnoresPath(remainingPath) {
		state, ok := s.local.RunningModels()[config.ComfyUIModelID]
		if !ok || state != process.StateReady {
			swaputil.SendResponse(w, r, http.StatusConflict,
				"model "+config.ComfyUIModelID+" is not loaded; ignored ComfyUI paths cannot start it")
			return
		}
	}

	*r = *r.WithContext(swaputil.SetContext(r.Context(), swaputil.ReqContextData{
		ApiKey:   swaputil.ExtractAPIKey(r),
		Model:    config.ComfyUIModelID,
		ModelID:  config.ComfyUIModelID,
		Metadata: make(map[string]string),
	}))
	// ComfyUI takes jobs on both paths: its frontend uses /api/prompt, many API
	// clients use /prompt.
	if r.Method == http.MethodPost && (remainingPath == "/api/prompt" || remainingPath == "/prompt") && s.metrics != nil {
		s.serveComfyUIPrompt(w, r)
		return
	}
	s.serveLocal(w, r)
}

// comfyUIPollInterval is how often a queued job's history is checked, and
// comfyUIJobMaxWait is how long a job may sit in the queue and run before the
// watcher gives up on it. Vars so tests can shorten them.
var (
	comfyUIPollInterval = 2 * time.Second
	comfyUIJobMaxWait   = 12 * time.Hour
)

// serveComfyUIPrompt forwards a job submission and files one activity row
// for it. ComfyUI answers POST /api/prompt as soon as the job is queued, so
// the row is written later by watchComfyUIJob, once the job has finished.
func (s *Server) serveComfyUIPrompt(w http.ResponseWriter, r *http.Request) {
	// The reply is a few bytes of JSON; keep it uncompressed so it can be read.
	r.Header.Del("Accept-Encoding")
	recorder := newBodyCopier(w)
	s.serveLocal(recorder, r)
	swaputil.MarkClientClosed(recorder, r)

	tm, _ := s.metrics.newEntry(config.ComfyUIModelID, hint.TaskImageGeneration, r, recorder)
	tm.ReqPath = "/comfyui" + r.URL.Path
	tm.Route = string(hint.RouteLocal)

	var queued struct {
		PromptID string `json:"prompt_id"`
	}
	switch {
	case recorder.Status() == swaputil.StatusClientClosedRequest:
		tm.ErrorMsg = "client disconnected before response"
	case recorder.Status() != http.StatusOK:
		tm.ErrorMsg = failedErrorMessage(recorder.Status(), recorder.body.Bytes(), nil)
	case json.Unmarshal(recorder.body.Bytes(), &queued) != nil || queued.PromptID == "":
		tm.ErrorMsg = "ComfyUI did not return a prompt_id"
	default:
		go s.watchComfyUIJob(queued.PromptID, tm, recorder.StartTime())
		return
	}
	s.fileComfyUIRow(tm)
}

// watchComfyUIJob polls ComfyUI's /history/<id> until the job finishes, then
// files its row. It talks to the model's proxy address directly, not through
// llama-swap, so polling never starts the model or resets its TTL. It stops
// without a row when llama-swap shuts down.
//
// ponytail: one goroutine and one poll per queued job; fine for a person
// clicking Queue, switch to one shared poll of /history if jobs pile up.
func (s *Server) watchComfyUIJob(promptID string, tm ActivityLogEntry, queuedAt time.Time) {
	historyURL := strings.TrimRight(s.cfg.Models[config.ComfyUIModelID].Proxy, "/") +
		"/history/" + url.PathEscape(promptID)
	client := &http.Client{Timeout: 10 * time.Second}
	ticker := time.NewTicker(comfyUIPollInterval)
	defer ticker.Stop()
	deadline := time.Now().Add(comfyUIJobMaxWait)

	for {
		select {
		case <-s.shutdownCtx.Done():
			return
		case <-ticker.C:
		}
		if state, ok := s.local.RunningModels()[config.ComfyUIModelID]; !ok || state != process.StateReady {
			tm.RespStatusCode = http.StatusServiceUnavailable
			tm.ErrorMsg = "model stopped before the job finished"
			break
		}
		if time.Now().After(deadline) {
			tm.RespStatusCode = http.StatusGatewayTimeout
			tm.ErrorMsg = "gave up waiting for the job to finish"
			break
		}
		job, ok := fetchComfyUIJob(s.shutdownCtx, client, historyURL, promptID)
		if !ok {
			continue // not finished yet, or ComfyUI was briefly unreachable
		}
		job.fill(&tm)
		break
	}
	if tm.DurationMs == 0 {
		tm.DurationMs = int(time.Since(queuedAt).Milliseconds())
	}
	tm.Timestamp = time.Now()
	s.fileComfyUIRow(tm)
}

func (s *Server) fileComfyUIRow(tm ActivityLogEntry) {
	if stored, ok := s.metrics.queueMetrics(tm); ok {
		s.metrics.emitMetric(stored)
	}
}

// comfyUIJob is one entry of ComfyUI's /history/<id> response.
type comfyUIJob struct {
	Status struct {
		StatusStr string `json:"status_str"`
		// Messages are [name, data] pairs, e.g.
		// ["execution_start", {"prompt_id": "...", "timestamp": 1700000000000}].
		Messages [][]json.RawMessage `json:"messages"`
	} `json:"status"`
	Outputs map[string]map[string]json.RawMessage `json:"outputs"`
}

// fetchComfyUIJob returns the job once ComfyUI reports it finished. The
// history entry only appears when the job is done, success or error.
func fetchComfyUIJob(ctx context.Context, client *http.Client, historyURL, promptID string) (comfyUIJob, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, historyURL, nil)
	if err != nil {
		return comfyUIJob{}, false
	}
	resp, err := client.Do(req)
	if err != nil {
		return comfyUIJob{}, false
	}
	defer resp.Body.Close()
	var history map[string]comfyUIJob
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&history) != nil {
		return comfyUIJob{}, false
	}
	job, ok := history[promptID]
	return job, ok && job.Status.StatusStr != ""
}

// fill copies a finished job's result into its activity row: the run time
// from ComfyUI's own timestamps, the error for a failed job, and how many
// files it produced.
func (j comfyUIJob) fill(tm *ActivityLogEntry) {
	var start, end int64
	for _, m := range j.Status.Messages {
		if len(m) != 2 {
			continue
		}
		var name string
		var data struct {
			Timestamp        int64  `json:"timestamp"`
			ExceptionMessage string `json:"exception_message"`
		}
		if json.Unmarshal(m[0], &name) != nil || json.Unmarshal(m[1], &data) != nil {
			continue
		}
		switch name {
		case "execution_start":
			start = data.Timestamp
		case "execution_success":
			end = data.Timestamp
		case "execution_error":
			end = data.Timestamp
			tm.RespStatusCode = http.StatusInternalServerError
			tm.ErrorMsg = strings.TrimSpace(data.ExceptionMessage)
		case "execution_interrupted":
			end = data.Timestamp
			tm.RespStatusCode = swaputil.StatusClientClosedRequest
			tm.ErrorMsg = "interrupted"
		}
	}
	if j.Status.StatusStr != "success" && tm.ErrorMsg == "" {
		tm.RespStatusCode = http.StatusInternalServerError
		tm.ErrorMsg = "job failed"
	}
	if start > 0 && end >= start {
		tm.DurationMs = int(end - start)
	}

	outputs := 0
	for _, node := range j.Outputs {
		for _, key := range []string{"images", "gifs", "videos", "audio"} {
			var files []json.RawMessage
			if json.Unmarshal(node[key], &files) == nil {
				outputs += len(files)
			}
		}
	}
	if tm.Metadata == nil {
		tm.Metadata = make(map[string]string)
	}
	tm.Metadata["outputs"] = strconv.Itoa(outputs)
}
