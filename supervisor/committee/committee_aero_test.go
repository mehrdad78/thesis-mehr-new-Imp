package committee

import (
	"blockEmulator/aero"
	"blockEmulator/params"
	"blockEmulator/supervisor/supervisor_log"
	"encoding/csv"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
)

func TestAERORewardLogsBaselineFeedbackAndFinalEpoch(t *testing.T) {
	oldPath, oldCfg := params.DataWrite_path, params.AERO
	defer func() { params.DataWrite_path = oldPath; params.AERO = oldCfg }()
	params.DataWrite_path = t.TempDir()
	params.AERO.OnlineTraining = true
	committee := &AEROCommittee{logger: &supervisor_log.SupervisorLog{Slog: log.New(io.Discard, "", 0)}}
	r := aero.RewardSummary{Reward: -2.32, CrossRatio: .8, CrossVariance: 1.5, NormalizedCrossVariance: .24, CrossWeight: 2, VarianceWeight: 3}
	committee.writeReward(0, r, 6, false)
	committee.writeReward(1, r, 7, false)
	committee.writeReward(2, r, 8, true)
	f, err := os.Open(filepath.Join(params.DataWrite_path, "aero_rewards.csv"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatal(rows)
	}
	if rows[1][1] != "" || rows[1][9] != "false" {
		t.Fatal("baseline assigned an action", rows[1])
	}
	if rows[2][0] != "1" || rows[2][1] != "1" || rows[2][9] != "true" {
		t.Fatal("feedback epoch wrong", rows[2])
	}
	if rows[3][9] != "false" || rows[3][10] != "true" {
		t.Fatal("final measurement incorrectly fed to PPO", rows[3])
	}
	committee.writeEpoch(aero.Snapshot{Epoch: 2}, aero.Plan{Reward: &r}, "digest")
	data, err := os.ReadFile(filepath.Join(params.DataWrite_path, "aero_epochs.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var entry struct {
		ObservedEpoch uint64             `json:"observed_epoch"`
		Reward        aero.RewardSummary `json:"reward"`
	}
	if err = json.Unmarshal(data, &entry); err != nil {
		t.Fatal(err)
	}
	if entry.ObservedEpoch != 1 || entry.Reward.Reward != -2.32 {
		t.Fatal(entry)
	}
}
