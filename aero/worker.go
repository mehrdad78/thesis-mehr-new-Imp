package aero

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

// Worker is a persistent Python CPU process. Each replica loads the same checkpoint.
type Worker struct {
	cmd    *exec.Cmd
	input  io.WriteCloser
	output *bufio.Reader
	mu     sync.Mutex
}

func StartWorker(c Config) (*Worker, error) {
	args := []string{"-u", "-m", "aero.worker", c.Checkpoint}
	if c.OnlineTraining {
		args = append(args, "--online")
	}
	cmd := exec.Command(c.Python, args...)
	cmd.Stderr = os.Stderr
	in, e := cmd.StdinPipe()
	if e != nil {
		return nil, e
	}
	out, e := cmd.StdoutPipe()
	if e != nil {
		return nil, e
	}
	if e = cmd.Start(); e != nil {
		return nil, e
	}
	return &Worker{cmd: cmd, input: in, output: bufio.NewReader(out)}, nil
}
func (w *Worker) Plan(s Snapshot) (Plan, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	b, e := json.Marshal(s)
	if e != nil {
		return Plan{}, e
	}
	if _, e = w.input.Write(append(b, '\n')); e != nil {
		return Plan{}, e
	}
	line, e := w.output.ReadBytes('\n')
	if e != nil {
		return Plan{}, fmt.Errorf("AERO Python worker: %w", e)
	}
	var answer struct {
		Moves       []Move         `json:"moves"`
		Error       string         `json:"error"`
		ModelDigest string         `json:"model_digest"`
		Reward      *RewardSummary `json:"reward"`
	}
	if e = json.Unmarshal(line, &answer); e != nil {
		return Plan{}, e
	}
	if answer.Error != "" {
		return Plan{}, fmt.Errorf("policy: %s", answer.Error)
	}
	p := Plan{Moves: answer.Moves, ModelDigest: answer.ModelDigest, Reward: answer.Reward}
	if p.Moves == nil {
		p.Moves = []Move{}
	}
	_, e = Expand(s, p)
	return p, e
}

// MeasureReward reads checkpoint weights even if a short run had no migration.
func (w *Worker) MeasureReward(stats Stats) (RewardSummary, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	b, e := json.Marshal(struct {
		Operation string `json:"operation"`
		Current   Stats  `json:"current"`
	}{"reward", stats})
	if e != nil {
		return RewardSummary{}, e
	}
	if _, e = w.input.Write(append(b, '\n')); e != nil {
		return RewardSummary{}, e
	}
	line, e := w.output.ReadBytes('\n')
	if e != nil {
		return RewardSummary{}, e
	}
	var answer struct {
		Reward *RewardSummary `json:"reward"`
		Error  string         `json:"error"`
	}
	if e = json.Unmarshal(line, &answer); e != nil {
		return RewardSummary{}, e
	}
	if answer.Error != "" || answer.Reward == nil {
		return RewardSummary{}, fmt.Errorf("reward measurement failed: %s", answer.Error)
	}
	return *answer.Reward, nil
}
func (w *Worker) Close() { w.input.Close(); w.cmd.Wait() }
