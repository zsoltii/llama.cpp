#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

struct llama_ubatch;
struct llama_vocab;

// reduced LM head for a draft (MTP) context: the draft only scores
//   - the static id prefix [0, n_static), read straight from the LM head rows, and
//   - the tokens seen in this context so far, copied into a small F16 head on the device
// all other tokens get -INF logits, so they are never drafted. the target model still
// scores the full vocabulary, so the output is unchanged; only the draft acceptance can drop
//
// a token is first scored after it was seen, so the first use of a word of an other language is never drafted.
// a seed text in the language of the chat fills the dynamic head with its tokens up front
//
// env: LLAMA_MTP_DRAFT_VOCAB=n_static (default 32768, -1 disables it), LLAMA_MTP_DRAFT_VOCAB_DYN=n_dyn (default 16384),
//      LLAMA_MTP_DRAFT_VOCAB_SEED=path of a text file
class llama_draft_vocab {
public:
    // rows of the dynamic head are scored in chunks of this size, so the graph only changes when a chunk fills up
    static constexpr int32_t ROW_CHUNK = 1024;

    llama_draft_vocab(const llama_vocab & vocab, int32_t n_static, int32_t n_dyn);

    // add the tokens of a text file, returns the number of tokens added
    int32_t add_text_file(const llama_vocab & vocab, const char * path);

    // returns nullptr if disabled by the env
    static std::unique_ptr<llama_draft_vocab> from_env(const llama_vocab & vocab);

    // allocate the dynamic head next to the LM head w, once
    bool init(const ggml_tensor * w);

    // assign dynamic rows to the new tokens of the ubatch, call before building the graph
    void add(const llama_ubatch & ubatch);

    // graph topology: number of dynamic rows scored and size of the row update
    int32_t n_rows() const;
    int32_t n_upd(const llama_ubatch & ubatch) const;

    // fill the graph inputs and consume the pending row updates
    void set_upd(int32_t * upd_tok, int32_t * upd_slot, int32_t n);
    void set_map(int32_t * map, int32_t n) const;

    ggml_backend_buffer_t buffer() const { return buf.get(); }

    int32_t n_vocab  = 0;
    int32_t n_static = 0;
    int32_t n_dyn    = 0;

    ggml_tensor * dyn = nullptr; // F16 [n_embd, n_dyn + 1], row n_dyn is a scratch row for the unused updates

private:
    std::vector<int32_t> slot; // [n_vocab] dynamic row of a token, -1 if none
    std::vector<int32_t> tok;  // [n_used] token of a dynamic row

    std::vector<std::pair<int32_t, int32_t>> pending; // (token, row) not copied to the device yet

    void add_token(int32_t t);

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
};
