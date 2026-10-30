#include "llama-draft-vocab.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-vocab.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

llama_draft_vocab::llama_draft_vocab(const llama_vocab & vocab, int32_t n_static, int32_t n_dyn)
    : n_vocab(vocab.n_tokens()), n_static(n_static), n_dyn(n_dyn) {
    slot.assign(n_vocab, -1);
    tok.reserve(n_dyn);

    // the control tokens (end of turn, </think>, tool calls, ...) are scored from the start
    for (int32_t t = n_static; t < n_vocab; ++t) {
        if (vocab.is_control(t)) {
            add_token(t);
        }
    }
}

std::unique_ptr<llama_draft_vocab> llama_draft_vocab::from_env(const llama_vocab & vocab) {
    const char * env_static = getenv("LLAMA_MTP_DRAFT_VOCAB");
    const char * env_dyn    = getenv("LLAMA_MTP_DRAFT_VOCAB_DYN");

    const int32_t n_vocab  = vocab.n_tokens();
    const int32_t n_static = env_static ? atoi(env_static) : 32768;
    const int32_t n_dyn    = std::max(ROW_CHUNK, env_dyn ? atoi(env_dyn) : 16384);

    if (n_static < 0 || n_static >= n_vocab) {
        return nullptr;
    }

    auto res = std::make_unique<llama_draft_vocab>(vocab, n_static, n_dyn);

    const char * env_seed = getenv("LLAMA_MTP_DRAFT_VOCAB_SEED");
    const int32_t n_seed = env_seed ? res->add_text_file(vocab, env_seed) : 0;

    LLAMA_LOG_INFO("%s: draft LM head scores token ids [0, %d) + up to %d tokens seen in the context (of %d), %d from the seed text\n",
            __func__, n_static, n_dyn, n_vocab, n_seed);

    return res;
}

int32_t llama_draft_vocab::add_text_file(const llama_vocab & vocab, const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LLAMA_LOG_WARN("%s: cannot read the seed text '%s'\n", __func__, path);
        return 0;
    }

    std::stringstream ss;
    ss << f.rdbuf();

    const size_t n_used0 = tok.size();
    for (const llama_token t : vocab.tokenize(ss.str(), /*add_special*/ false, /*parse_special*/ false)) {
        add_token(t);
    }

    return (int32_t) (tok.size() - n_used0);
}

bool llama_draft_vocab::init(const ggml_tensor * w) {
    if (dyn) {
        return true;
    }

    if (!w->buffer || w->ne[1] != n_vocab) {
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(params));

    dyn = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, w->ne[0], n_dyn + 1);
    ggml_set_name(dyn, "draft_vocab_dyn");

    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_buffer_get_type(w->buffer)));
    if (!buf) {
        LLAMA_LOG_WARN("%s: failed to allocate the draft LM head, using the full vocabulary\n", __func__);
        dyn = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(buf.get(), 0);
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    LLAMA_LOG_INFO("%s: %s draft LM head buffer size = %.2f MiB\n", __func__,
            ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);

    return true;
}

void llama_draft_vocab::add_token(int32_t t) {
    if (t < n_static || t >= n_vocab || slot[t] >= 0 || (int32_t) tok.size() >= n_dyn) {
        return;
    }

    slot[t] = (int32_t) tok.size();
    tok.push_back(t);

    pending.emplace_back(t, slot[t]);
}

void llama_draft_vocab::add(const llama_ubatch & ubatch) {
    if (!ubatch.token) {
        return;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        add_token(ubatch.token[i]);
    }
}

int32_t llama_draft_vocab::n_rows() const {
    const int32_t n = ((int32_t) tok.size() + ROW_CHUNK - 1)/ROW_CHUNK*ROW_CHUNK;

    return std::min(n_dyn, std::max(ROW_CHUNK, n));
}

int32_t llama_draft_vocab::n_upd(const llama_ubatch & ubatch) const {
    return std::max((int32_t) ubatch.n_tokens, (int32_t) pending.size());
}

void llama_draft_vocab::set_upd(int32_t * upd_tok, int32_t * upd_slot, int32_t n) {
    GGML_ASSERT((int32_t) pending.size() <= n);

    for (int32_t i = 0; i < n; ++i) {
        if (i < (int32_t) pending.size()) {
            upd_tok [i] = pending[i].first;
            upd_slot[i] = pending[i].second;
        } else {
            // the unused updates copy row 0 into the scratch row
            upd_tok [i] = 0;
            upd_slot[i] = n_dyn;
        }
    }

    pending.clear();
}

void llama_draft_vocab::set_map(int32_t * map, int32_t n) const {
    // the unused rows land on the logit past the end of the vocabulary, which is dropped
    for (int32_t i = 0; i < n; ++i) {
        map[i] = i < (int32_t) tok.size() ? tok[i] : n_vocab;
    }
}
