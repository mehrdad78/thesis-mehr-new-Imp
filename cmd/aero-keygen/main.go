// Generate local emulator identities and a pinned public-key manifest.
package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
)

func main() {
	dir := flag.String("dir", "aero/keys", "key directory")
	shards := flag.Int("shards", 4, "physical shards")
	nodes := flag.Int("nodes", 4, "nodes per shard")
	flag.Parse()
	if *shards < 1 || *nodes < 1 {
		log.Fatal("positive shard/node counts required")
	}
	if _, e := os.Stat(*dir); !os.IsNotExist(e) {
		log.Fatal("key directory already exists; use a new directory")
	}
	if e := os.MkdirAll(*dir, 0700); e != nil {
		log.Fatal(e)
	}
	public := make(map[uint64][]byte)
	for id := 0; id <= *shards**nodes; id++ {
		pub, private, e := ed25519.GenerateKey(rand.Reader)
		if e != nil {
			log.Fatal(e)
		}
		public[uint64(id)] = pub
		if e = os.WriteFile(filepath.Join(*dir, strconv.Itoa(id)+".key"), private, 0600); e != nil {
			log.Fatal(e)
		}
	}
	data, _ := json.MarshalIndent(public, "", "  ")
	if e := os.WriteFile(filepath.Join(*dir, "public.json"), data, 0644); e != nil {
		log.Fatal(e)
	}
	fmt.Printf("Generated %d node keys and one supervisor key in %s\n", *shards**nodes, *dir)
}
