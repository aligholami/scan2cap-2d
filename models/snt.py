import torch
import torch.nn as nn
import random


class CaptionBase(nn.Module):
    def __init__(self,
                 device,
                 max_desc_len,
                 vocabulary,
                 embeddings,
                 emb_size,
                 feat_size,
                 hidden_size
                 ):
        super().__init__()

        self.device = device
        self.max_desc_len = max_desc_len
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])
        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU()
        )
        self.recurrent_cell = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)

    def step(self, step_input, hidden):
        hidden = self.recurrent_cell(step_input, hidden)

        return hidden, hidden


class ShowAndTell(CaptionBase):
    def __init__(self,
                 device,
                 max_desc_len,
                 training_tf,
                 vocabulary,
                 embeddings,
                 emb_size,
                 feat_size,
                 feat_input,
                 hidden_size
                 ):

        super().__init__(
            device=device,
            max_desc_len=max_desc_len,
            vocabulary=vocabulary,
            embeddings=embeddings,
            emb_size=emb_size,
            feat_size=feat_size,
            hidden_size=hidden_size
        )
        self.feat_input = feat_input
        assert self.feat_input['add_global']
        self.training_tf = training_tf

    def forward(self, data_dict, use_tf=True, is_eval=False):

        if self.feat_input['add_target']:
            g_feat = data_dict["g_feat"]
            t_feat = data_dict["t_feat"]
            t_feat = torch.cat((g_feat, t_feat), dim=1)
            data_dict['t_feat'] = t_feat

        if not is_eval:
            # During training
            data_dict = self.forward_sample_batch(data_dict)
        else:
            # During evaluation
            use_tf = False
            data_dict = self.forward_scene_batch(data_dict, use_tf)

        return data_dict

    def forward_sample_batch(self, data_dict):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        t_feat = data_dict["t_feat"]

        num_words = des_lens[0]
        batch_size = des_lens.shape[0]
        t_feat = self.map_feat(t_feat.squeeze())

        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = t_feat  # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id, :]  # batch_size, emb_size
        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

            # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.tensor(self.embeddings[word], dtype=torch.float).unsqueeze(0).to(self.device)
                step_preds.append(emb)

            step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)

            # next step
            step_id += 1
            if step_id == num_words - 1: break  # exit for train mode

            use_tf = False if random.random() > self.training_tf else True
            step_input = step_preds if not use_tf else word_embs[:, step_id]  # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict

    def forward_scene_batch(self, data_dict, use_tf=False):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        t_feat = data_dict["t_feat"]
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]

        # transform the features
        t_feat = self.map_feat(t_feat.squeeze())  # batch_size, num_proposals, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []

        # start recurrence
        hidden = t_feat  # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id]  # batch_size, emb_size

        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

            # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.tensor(self.embeddings[word], dtype=torch.float).unsqueeze(0).to(self.device)
                step_preds.append(emb)

            step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs

            # next step
            step_id += 1
            if not use_tf and step_id == self.max_desc_len - 1: break  # exit for eval mode
            if use_tf and step_id == num_words - 1: break  # exit for train mode
            step_input = step_preds if not use_tf else word_embs[:, step_id]  # batch_size, emb_size

            outputs.append(step_output)

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict
