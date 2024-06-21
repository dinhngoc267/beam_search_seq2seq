import tqdm
from heapq import heappush, heappop


class BeamSearchNode(object):
    def __init__(self,
                 hidden_state,
                 prev_node,
                 token_id,
                 log_p,
                 length,
                 attn_buffer=None):
        if attn_buffer is None:
            attn_buffer = []

        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_p = log_p
        self.length = length
        self.attn_buffer = attn_buffer

    def eval(self):
        return self.log_p / float(self.length - 1 + 1e-6)


class PointerDecoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_layer,
                 embedding_dim,
                 hidden_size,
                 n_layers,
                 dropout,
                 device):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding_layer = embedding_layer

        self.attn = BahdanauAttention(key_dim=hidden_size * 2,
                                      query_dim=hidden_size * 2,
                                      hidden_dim=hidden_size)

        self.lstm = nn.GRU(input_size=embedding_dim,
                           hidden_size=hidden_size * 2,
                           num_layers=1,  # n_layers,
                           dropout=0,  # dropout if n_layers > 1 else 0,
                           bidirectional=False,
                           batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.Wo = nn.Linear(in_features=hidden_size * 2 + hidden_size * 2,
                            out_features=vocab_size)

        self.We = nn.Linear(in_features=hidden_size * 2,
                            out_features=hidden_size * 2)

        self.Wc = nn.Linear(in_features=hidden_size * 2,
                            out_features=1)
        self.Ws = nn.Linear(in_features=hidden_size * 2,
                            out_features=1)
        self.Wx = nn.Linear(in_features=embedding_dim,
                            out_features=1)

        self.Wg = nn.Linear(in_features=hidden_size * 2 + hidden_size * 2 + embedding_dim,
                            out_features=1)

        self.device = device

    def one_step_forward(self,
                         encoder_outputs,
                         decoder_input,
                         coverage,
                         previous_decoder_hidden_state,
                         extended_source_ids,
                         len_src_oovs):
        """Run one step"""

        batch_size = encoder_outputs.size(0)
        input_embedded = self.dropout(self.embedding_layer(decoder_input))

        # 1. update decoder hidden state first based on input and previous hidden state
        decoder_output, decoder_hidden_state = self.lstm(input_embedded, previous_decoder_hidden_state)
        # print(decoder_output.shape)
        # 2.compute context vector and attn scores based on encoder output and last hidden state
        # use decoder_output as query to compute context vector
        # last_decoder_hidden_state = decoder_hidden_state.permute(1, 0, 2)
        # last_decoder_hidden_state = decoder_hidden_state[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)
        last_decoder_hidden_state = decoder_hidden_state[-1].unsqueeze(0).permute(1, 0, 2)
        context_vec, att_scores = self.attn(encoder_outputs,
                                            last_decoder_hidden_state,
                                            coverage,
                                            extended_source_ids)

        # 3. compute p_vocab based on last decoder hidden state and context vec
        context_vec = context_vec.squeeze(1)

        p_vocab = F.softmax(self.Wo(torch.cat([decoder_output.squeeze(1),
                                               context_vec],
                                              dim=-1)),
                            dim=-1)

        # 4. compute probability of generating token from vocab
        p_gen = F.sigmoid(self.Wc(context_vec) +
                          self.Ws(last_decoder_hidden_state.squeeze(1)) +
                          self.Wx(input_embedded).squeeze(1))

        # 5. compute final probs by combining attention scores with p_vocab
        max_num_src_oov = max(len_src_oovs)
        p_copy = torch.zeros((batch_size, self.vocab_size + max_num_src_oov + 1),
                             dtype=torch.float,
                             device=self.device)

        # projecting the attn score into the final probs
        att_scores = att_scores.squeeze(1)
        p_copy = p_copy.scatter_reduce_(1,
                                        extended_source_ids.clip(0, p_copy.size(1) - 1),
                                        att_scores,
                                        reduce='amax')[:, :-1]

        # print(p_gen)
        # extend the p_vocab vector
        p_extended_vocab = torch.zeros_like(p_copy, device=self.device)
        p_extended_vocab[:, :p_vocab.size(1)] = p_vocab

        p_final = p_gen * p_extended_vocab + (1 - p_gen) * p_copy

        # apply log function
        p_final = torch.log(p_final + 1e-9)

        return p_final.unsqueeze(1), att_scores, decoder_hidden_state, p_gen

    @staticmethod
    def compute_coverage(attn_scores):
        if len(attn_scores) == 0:
            return None
        return torch.sum(torch.cat(attn_scores, dim=1), dim=1)

        # attn_scores = torch.cat(attn_scores, dim =1)
        # batch_size = attn_scores.size(0)
        #
        # attn_scores = attn_scores.view(-1, attn_scores.size(-1))
        # _, max_attn_indices = torch.max(attn_scores, dim=-1)
        #
        # mask = torch.nn.functional.one_hot(max_attn_indices, attn_scores.size(-1))
        # coverage = mask*attn_scores # batch_size*time_step, source_len
        # coverage = coverage.view(batch_size, -1, coverage.size(-1)) # batch_Sze, timestep, src_len
        #
        # max_attn_coverage = torch.sum(coverage, dim=1) # batch_sizee, timestep
        # # print(coverage.shape)
        # # print(coverage)
        # coverage =  torch.sum(attn_scores.view(batch_size, -1, attn_scores.size(-1)), dim=1)
        # return max_attn_coverage, coverage
        # print(max_attn_indices.shape)

        # return torch.sum(torch.cat(attn_scores, dim=1), dim=1)
        # coverage = torch.cat(attn_scores, dim=1)
        # coverage = torch.sum(coverage, dim=1)
        # _, indices = attn_scores.max(dim=-1)
        # torch.nn.functional.one_hot(indices, n)
        # return coverage

    def forward(self,
                extended_source_ids,
                encoder_outputs,
                len_src_oovs,
                encoder_hidden_state,
                sos_id,
                unk_id,
                max_len,
                dec_inp_ids=None,
                use_coverage=True):

        batch_size = encoder_outputs.size(0)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(sos_id)

        p_finals = []
        cov_losses = []
        attn_scores_buffer = []

        # coverage = torch.zeros(batch_size, encoder_outputs.size(1), device=device)
        coverage = None
        max_attn_coverage = None

        p_gen_buffer = []

        if dec_inp_ids is not None:
            max_len = dec_inp_ids.shape[1]
        for i in range(max_len):
            p_final, attn_scores, decoder_hidden_state, p_gen = self.one_step_forward(encoder_outputs=encoder_outputs,
                                                                                      decoder_input=decoder_input,
                                                                                      coverage=coverage,
                                                                                      previous_decoder_hidden_state=decoder_hidden_state,
                                                                                      extended_source_ids=extended_source_ids,
                                                                                      len_src_oovs=len_src_oovs)

            p_finals.append(p_final)
            p_gen_buffer.append(p_gen)
            p_teacher_forcing = 1  # random.random()
            if dec_inp_ids is not None:  # and p_teacher_forcing > 0.2:  # 0.8 chance of doing teacher forcing
                decoder_input = dec_inp_ids[:, i].unsqueeze(1)
            else:  # inference mode
                _, topi = p_final.topk(1, largest=True, dim=-1)
                decoder_input = topi.detach().squeeze(1)  # [batch_size, 1]
                # as some ids might be temporary ids from source ids. these id must be converted to unk ids
                decoder_input[decoder_input >= self.vocab_size] = unk_id

            if len(attn_scores_buffer) > 0:
                coverage = self.compute_coverage(attn_scores_buffer)
            if max_attn_coverage is not None:
                cov_loss = torch.min(max_attn_coverage, attn_scores)
                cov_loss = torch.sum(cov_loss, dim=-1)
                cov_losses.append(cov_loss)

            attn_scores_buffer.append(attn_scores.unsqueeze(1))
        p_gen_buffer = torch.stack(p_gen_buffer)
        p_gen_buffer = p_gen_buffer.view(-1)
        p_finals = torch.cat(p_finals, dim=1)
        if cov_losses:
            cov_losses = torch.cat(cov_losses)

        return p_finals, cov_losses, p_gen_buffer

    def beam_decode(self,
                    extended_source_ids,
                    encoder_outputs,
                    len_src_oovs,
                    encoder_hidden_state,
                    sos_id,
                    unk_id,
                    eos_id,
                    max_len,
                    topk=3,
                    use_coverage=True):

        batch_output_sequences = []
        batch_size = encoder_outputs.size(0)

        for i in range(batch_size):
            # assign last encoder hidden state to decoder hidden state
            decoder_hidden_state = encoder_hidden_state[:, i, :].unsqueeze(0)
            encoder_output = encoder_outputs[i, :, :].unsqueeze(0)

            # starting node
            start_node = BeamSearchNode(hidden_state=decoder_hidden_state,
                                        prev_node=None,
                                        token_id=sos_id,
                                        log_p=0,
                                        length=1)

            nodes = []
            end_nodes = []
            # start the queue
            heappush(nodes, (-start_node.eval(), id(start_node), start_node))
            # heappush(step_t_nodes, (-start_node.eval(), id(start_node), start_node))

            # a flag to know when already get enough top k sequences
            flag = False
            step = 0
            # give up when decoding takes too long
            while step < 300:
                step_t_nodes = []
                # renew end_nodes because the current end nodes are in nodes
                end_nodes = []
                while len(nodes) != 0:
                    # fetch the best node
                    score, _, current_node = heappop(nodes)
                    token_id = current_node.token_id
                    if token_id >= self.vocab_size:
                        token_id = unk_id

                    decoder_input = torch.tensor([token_id], dtype=torch.long, device=self.device).unsqueeze(0)
                    decoder_hidden_state = current_node.hidden_state
                    attn_buffer = current_node.attn_buffer

                    if current_node.token_id == eos_id and current_node.prev_node is not None:  # reach eos token
                        end_nodes.append((score, id(current_node), current_node))
                        # if we reach maximum # of sequences required
                        if len(end_nodes) >= topk:
                            flag = True
                            break
                        else:
                            continue

                    # decode one step using decoder
                    coverage = self.compute_coverage(attn_buffer)

                    p_final, att_scores, decoder_hidden_state, p_gen = self.one_step_forward(
                        encoder_outputs=encoder_output,
                        decoder_input=decoder_input,
                        previous_decoder_hidden_state=decoder_hidden_state,
                        extended_source_ids=extended_source_ids[i, :].unsqueeze(0),
                        len_src_oovs=[len_src_oovs[i]],
                        coverage=coverage,
                    )

                    attn_buffer.append(att_scores.unsqueeze(1))

                    # compute next topk tokens
                    topk_log_p, topk_indices = p_final.topk(topk)

                    topk_log_p = topk_log_p.flatten().cpu().numpy()
                    topk_indices = topk_indices.flatten().cpu().numpy()

                    for k in range(topk):
                        node = BeamSearchNode(hidden_state=copy.copy(decoder_hidden_state),
                                              prev_node=current_node,
                                              token_id=topk_indices[k],
                                              log_p=topk_log_p[k] + current_node.log_p,
                                              length=current_node.length + 1,
                                              attn_buffer=copy.copy(attn_buffer))

                        # heappush(nodes, (-node.eval(), id(node), node))
                        heappush(step_t_nodes, (-node.eval(), id(node), node))

                if flag:
                    break

                # extend current end nodes to compare scores
                if len(end_nodes) > 0:
                    step_t_nodes.extend(end_nodes)
                # sort by score of top nodes at step t. nodes always have the past end nodes.
                # it means if nodes contains one end node which is <eos> then in the next while loop,
                # it only runs forward to find topk - 1 end nodes.
                nodes = sorted(step_t_nodes, key=lambda x: x[0])[:topk]
                step += 1

            # if len(end_nodes) == 0:
            #     # end_nodes = [heappop(nodes) for _ in range(topk)]
            #     end_nodes = nodes
            if len(end_nodes) < topk:
                end_nodes = nodes

            # here I only return top 1 sequence,
            # remove the break and use a buffer to save top k output sequences of each input sequence
            for score, _id, node in sorted(end_nodes, key=lambda x: x[0]):
                sequence = [node.token_id]

                while node.prev_node is not None:
                    node = node.prev_node
                    sequence.append(node.token_id)
                sequence = sequence[::-1]  # reverse
                batch_output_sequences.append(sequence)
                break

        return batch_output_sequences
