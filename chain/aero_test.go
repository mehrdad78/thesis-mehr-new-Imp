package chain

import (
	"blockEmulator/aero"
	"blockEmulator/core"
	"blockEmulator/params"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/trie"
	"math/big"
	"path/filepath"
	"testing"
)

func TestAEROMigrationPreservesStateAndDeletesSource(t *testing.T) {
	oldMethod, oldShards, oldPath, oldCfg := params.ConsensusMethod, params.ShardNum, params.DatabaseWrite_path, params.AERO
	defer func() {
		params.ConsensusMethod = oldMethod
		params.ShardNum = oldShards
		params.DatabaseWrite_path = oldPath
		params.AERO = oldCfg
	}()
	params.ConsensusMethod = 4
	params.ShardNum = 2
	params.AERO.PrefixDigits = 1
	params.DatabaseWrite_path = filepath.ToSlash(t.TempDir()) + "/"
	source, e := NewBlockChain(&params.ChainConfig{ShardID: 0, ShardNums: 2, Nodes_perShard: 4}, rawdb.NewMemoryDatabase())
	if e != nil {
		t.Fatal(e)
	}
	defer source.CloseBlockChain()
	dest, e := NewBlockChain(&params.ChainConfig{ShardID: 1, ShardNums: 2, Nodes_perShard: 4}, rawdb.NewMemoryDatabase())
	if e != nil {
		t.Fatal(e)
	}
	defer dest.CloseBlockChain()
	address := "a000000000000000000000000000000000000000"
	state := &core.AccountState{Nonce: 91, Balance: big.NewInt(12345)}
	source.AddAccounts([]string{address}, []*core.AccountState{state}, 0)
	fetched := source.FetchAccounts([]string{address})
	for _, bc := range []*BlockChain{source, dest} {
		bc.Update_PartitionMap(address, 1)
		bc.Update_PartitionMap(aero.PrefixKey(10, 1), 1)
	}
	source.AddAccounts(nil, nil, 0)
	dest.AddAccounts([]string{address}, fetched, 0)
	got := dest.FetchAccounts([]string{address})[0]
	if got.Nonce != 91 || got.Balance.Cmp(big.NewInt(12345)) != 0 {
		t.Fatal("state lost")
	}
	st, e := trie.New(trie.TrieID(common.BytesToHash(source.CurrentBlock.Header.StateRoot)), source.triedb)
	if e != nil {
		t.Fatal(e)
	}
	value, e := st.Get([]byte(address))
	if e != nil || value != nil {
		t.Fatal("source state retained", e)
	}
	if dest.Get_PartitionMap("afffffffffffffffffffffffffffffffffffffff") != 1 {
		t.Fatal("future account not routed by prefix")
	}
	// Move the group back: migrated balances/nonces must survive repeated movement.
	fetched = dest.FetchAccounts([]string{address})
	for _, bc := range []*BlockChain{source, dest} {
		bc.Update_PartitionMap(address, 0)
		bc.Update_PartitionMap(aero.PrefixKey(10, 1), 0)
	}
	dest.AddAccounts(nil, nil, 0)
	source.AddAccounts([]string{address}, fetched, 0)
	got = source.FetchAccounts([]string{address})[0]
	if got.Nonce != 91 || got.Balance.Cmp(big.NewInt(12345)) != 0 {
		t.Fatal("return migration changed state")
	}
}
