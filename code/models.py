import numpy as np
import torch
import torch.nn.functional as F

from layers import FeaturesEmbedding
from layers import Linear
from layers import FactorizationLayer
from layers import MultiLayerPerceptron
from layers import InnerProductNetwork
from layers import FieldAwareFactorizationLayer
from layers import CrossNet
from layers import HistAtt, HistAtt_S,CoAtt,HistAtt2,CoAtt2
from layers import PLELayer
from layers import MetaUnit,MetaAttention

class Rec(torch.nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.vocabulary_size = data_config['vocabulary_size']
        self.user_num_fields = data_config['user_num_fields']
        self.item_num_fields = data_config['item_num_fields']
        self.domain_num_fields = data_config['domain_num_fields']

        self.num_fields = data_config['num_fields']
        self.embed_dim = model_config['embed_dim'] if 'embed_dim' in model_config else None
        self.hidden_dims = model_config['hidden_dims'] if 'hidden_dims' in model_config else None
        self.tower_dims = model_config['tower_dims'] if 'tower_dims' in model_config else None
        self.gru_h_dim = model_config['gru_h_dim'] if 'gru_h_dim' in model_config else None
        self.dropout = model_config['dropout'] if 'dropout' in model_config else None
        self.use_hist = model_config['use_hist'] if 'use_hist' in model_config else None
        self.have_multi_feature = model_config['have_multi_feature'] if 'have_multi_feature' in model_config else None
        self.batch_random_neg = model_config['batch_random_neg'] if 'batch_random_neg' in model_config else None
        self.lamda1 = model_config['lamda1'] if 'lamda1' in model_config else None
        self.max_session_len = data_config['max_session_len'] if 'max_session_len' in data_config else None
        self.max_hist_len = data_config['max_hist_len'] if 'max_hist_len' in data_config else None

        self.hist_session_num = model_config['hist_session_num'] if 'hist_session_num' in model_config else None
        self.hist_session_length = model_config['hist_session_length'] if 'hist_session_length' in model_config else None
        self.n_heads = model_config['n_heads'] if 'n_heads' in model_config else None
        self.expert_num = model_config['expert_num'] if 'expert_num' in model_config else None
        self.task_num = model_config['task_num'] if 'task_num' in model_config else None
        self.domain_num = data_config['domain_num'] if 'domain_num' in data_config else None
        self.domain_size = data_config['domain_size'] if 'domain_size' in data_config else None
        self.meta_dims = model_config['meta_dims'] if 'meta_dims' in model_config else None

class LR(Rec):
    """
    A pytorch implementation of Logistic Regression.
    """
    def __init__(self, model_config, data_config):
        super(LR, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
    
    def get_name(self):
        return 'LR'

    def forward(self, x_user, x_item, x_domain,user_hist = None, hist_len = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item,x_domain), dim=1)
        return torch.sigmoid(self.linear(x).squeeze(1)),None


class DSSM(Rec):
    """
    A pytorch implementation of DSSM as recall model, plain dual tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(DSSM, self).__init__(model_config, data_config)
        self.user_vec_dim = self.user_num_fields * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.user_tower = MultiLayerPerceptron(self.user_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')
        self.item_tower = MultiLayerPerceptron(self.item_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')

    def get_name(self):
        return 'DSSM'

    def forward(self, x_user, x_item, x_stat, user_hist = None, hist_len = None):
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        self.u = self.user_tower(self.user_emb)
        self.i = self.item_tower(self.item_emb)
        score = torch.sigmoid(torch.sum(self.u * self.i, dim=1, keepdim=False))
        # score = torch.sigmoid(torch.cosine_similarity(self.u, self.i, dim=1))
        return score

    def get_user_repre(self, x_user, user_hist = None, hist_len = None):
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        return self.user_tower(self.user_emb)

    def get_item_repre(self, x_item):
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        return self.item_tower(self.item_emb)


class COLD(Rec):
    """
    A pytorch implementation of COLD as pre-ranking model, plain triple tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(COLD, self).__init__(model_config, data_config)
        self.user_vec_dim = self.user_num_fields * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim
        self.cross_vec_dim = self.num_fields*(self.num_fields-1)/2


        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.all_tower = MultiLayerPerceptron(self.user_vec_dim+self.item_vec_dim+self.cross_vec_dim, self.hidden_dims,self.dropout, True)


    def get_name(self):
        return 'COLD'

    def forward(self, x_user, x_item, x_stat, user_hist = None, hist_len = None, ubr_hist = None, ubr_hist_len=None):
        x = torch.cat((x_user, x_item, x_stat), dim=1)
        self.user_emb = self.embedding(x_user).view(-1, self.user_vec_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        self.all_emb = self.embedding(x)
        self.cross_emb = self.inner_product(self.all_emb)

        self.all_emb = torch.cat((self.user_emb,self.item_emb,self.cross_emb),1)

        self.a = self.all_tower(self.all_emb)

        return self.a

class YouTubeDNN(Rec):
    """
    A pytorch implementation of DSSM as recall model, plain dual tower DNN.
    """

    def __init__(self, model_config, data_config):
        super(YouTubeDNN, self).__init__(model_config, data_config)
        self.user_vec_dim = (self.user_num_fields + self.item_num_fields) * self.embed_dim
        self.item_vec_dim = self.item_num_fields * self.embed_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.user_tower = MultiLayerPerceptron(self.user_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')
        self.item_tower = MultiLayerPerceptron(self.item_vec_dim, self.hidden_dims, self.dropout, False, 'tanh')

    def get_name(self):
        return 'YouTubeDNN'

    def forward(self, x_user, x_item, x_stat, user_hist, hist_len):
        self.user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        
        self.user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        # get mask
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        self.user_hist[~mask] = 0.0

        self.user_hist_rep = torch.mean(self.user_hist, dim=1)
        self.u = self.user_tower(torch.cat((self.user_emb, self.user_hist_rep), dim=1))
        self.i = self.item_tower(self.item_emb)
        score = torch.sigmoid(torch.sum(self.u * self.i, dim=1, keepdim=False))
        # score = torch.sigmoid(torch.cosine_similarity(self.u, self.i, dim=1))
        return score

    def get_user_repre(self, x_user, user_hist, hist_len):
        self.user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        self.user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        # get mask
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        self.user_hist[~mask] = 0.0
        
        self.user_hist_rep = torch.mean(self.user_hist, dim=1)
        return self.user_tower(torch.cat((self.user_emb, self.user_hist_rep), dim=1))

    def get_item_repre(self, x_item):
        self.item_emb = self.embedding(x_item).view(-1, self.item_vec_dim)
        return self.item_tower(self.item_emb)


class WDL(Rec):
    """
    A pytorch implementation of wide and deep learning.
    """

    def __init__(self, model_config, data_config):
        super(WDL, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)
    
    def get_name(self):
        return 'WDL'
    
    def forward(self, x_user, x_item, user_hist = None, hist_len = None, ubr_hist = None, ubr_hist_len=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.input_dim))
        return torch.sigmoid(x.squeeze(1))

class FM(Rec):
    def __init__(self, model_config, data_config):
        super(FM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
    
    def get_name(self):
        return 'FM'
    
    def forward(self, x_user, x_item, x_domain,user_hist, hist_len):
        x = torch.cat((x_user, x_item,x_domain), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x)
        return torch.sigmoid(x.squeeze(1)),None

class FFM(Rec):
    def __init__(self, model_config, data_config):
        super(FFM,self).__init__(model_config,data_config)
        self.linear = Linear(self.vocabulary_size)
        self.ffm = FieldAwareFactorizationLayer(self.vocabulary_size,self.num_fields,self.embed_dim)

    def get_name(self):
        return 'FFM'

    def forward(self, x_user, x_item, user_hist = None, hist_len = None, ubr_hist = None, ubr_hist_len=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x_user, x_item),dim=1)
        x = self.linear(x) + self.ffm(x)
        return torch.sigmoid(x.squeeze(1))

class PNN(Rec):
    def __init__(self,model_config,data_config):
        super(PNN,self).__init__(model_config,data_config)
        self.d1 = self.hidden_dims[0]
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.linear_z = torch.nn.Linear(in_features=self.num_fields * self.embed_dim, out_features=self.d1)
        torch.nn.init.xavier_uniform_(self.linear_z.weight)
        self.inner_product = InnerProductNetwork()
        self.linear_p = torch.nn.Linear(in_features=int(self.num_fields*(self.num_fields-1)/2), out_features=self.d1)
        torch.nn.init.xavier_uniform_(self.linear_p.weight)
        self.bias = torch.nn.Parameter(torch.randn(self.d1))
        self.mlp = MultiLayerPerceptron(self.d1,self.hidden_dims,self.dropout)
        self.l1_layer = torch.nn.ReLU()


    def get_name(self):
        return 'PNN'
    
    def forward(self, x_user, x_item, user_hist = None, hist_len = None, ubr_hist = None, ubr_hist_len=None):
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        #compute l_z
        z = embed_x.view(-1,self.num_fields*self.embed_dim)
        l_z = self.linear_z(z)

        #compute l_p
        p = self.inner_product(embed_x)
        l_p = self.linear_p(p)

        #mlp layers
        l1_in = torch.add(l_z,l_p)
        l1_in = torch.add(l1_in, self.bias)
        l1_out = self.l1_layer(l1_in)
        y = self.mlp(l1_out)
        return torch.sigmoid(y.squeeze(1))


class DeepFM(Rec):
    """
    A pytorch implementation of DeepFM.
    """
    def __init__(self, model_config, data_config):
        super(DeepFM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)

    def get_name(self):
        return 'DeepFM'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        x = torch.cat((x_user, x_item), dim=1)
        
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.input_dim))
        return torch.sigmoid(x.squeeze(1)),None

class DCN(Rec):
    """
    A pytorch implementation of DCN.
    """
    def __init__(self, model_config, data_config):
        super(DCN, self).__init__(model_config, data_config)
        self.input_dim = self.num_fields * self.embed_dim
        self.cross_net = CrossNet(self.input_dim, 5)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout,output_layer=False)
        final_dim =self.input_dim+self.hidden_dims[-1]
        self.fc = torch.nn.Linear(final_dim,1)

    def get_name(self):
        return 'DCN'

    def forward(self, x_user, x_item, x_domain, theme_hist, theme_hist_len, user_hist, hist_len):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        embed_x = self.embedding(x)
        cross_out = self.cross_net(embed_x.view(-1,self.input_dim))
        dnn_out = self.mlp(embed_x.view(-1,self.input_dim))
        final_out = torch.cat([cross_out,dnn_out],dim=-1)
        return torch.sigmoid(self.fc(final_out).squeeze(1)),None


class DCN_DEBIAS(Rec):
    """
    A pytorch implementation of DCN_DEBIAS.
    """
    def __init__(self, model_config, data_config):
        super(DCN_DEBIAS, self).__init__(model_config, data_config)
        self.input_dim = self.num_fields * self.embed_dim
        self.cross_net = CrossNet(self.input_dim, 5)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout,output_layer=False)
        self.mlp2_dim = self.domain_num_fields*self.embed_dim
        self.mlp2 = MultiLayerPerceptron(self.mlp2_dim,self.hidden_dims,self.dropout)
        final_dim =self.input_dim+self.hidden_dims[-1]
        self.fc = torch.nn.Linear(final_dim,1)

    def get_name(self):
        return 'DCN_DEBIAS'

    def forward(self, x_user, x_item, x_domain, user_hist = None, hist_len = None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        embed_x = self.embedding(x)
        d_inp = self.embedding(x_domain).view(-1,self.mlp2_dim)
        
        y_d = self.mlp2(d_inp).squeeze(1)
        cross_out = self.cross_net(embed_x.view(-1,self.input_dim))
        dnn_out = self.mlp(embed_x.view(-1,self.input_dim))
        final_out = torch.cat([cross_out,dnn_out],dim=-1)
        y_uid = self.fc(final_out).squeeze(1)
        return y_uid*torch.sigmoid(y_d),y_d


class DIN(Rec):
    def __init__(self, model_config, data_config):
        super(DIN, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.label_embedding = FeaturesEmbedding(2, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt(self.item_num_fields * self.embed_dim)
        self.att2 = HistAtt2(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        
    def get_name(self):
        return 'DIN'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist_click = theme_hist[:,:,[-1]]
        theme_hist = theme_hist[:,:,:-1]
        user_hist_click = user_hist[:,:,[-1]]
        user_hist = user_hist[:,:,:-1]
      
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist_click = self.label_embedding(user_hist_click).view(-1,user_hist_click.shape[1],user_hist_click.shape[2]*self.embed_dim)

        x = torch.cat((x_user, x_item,x_domain), dim=1)
        
        embed_x = self.embedding(x)
   
        # user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        theme_hist = self.embedding(theme_hist).view(-1,theme_hist.shape[1], theme_hist.shape[2] * self.embed_dim)
        # user_rep, score = self.att(item_emb,user_hist, hist_len)
        theme_hist_rep, score = self.att(item_emb, theme_hist ,hist_len)
        # user_rep, label_rep, score = self.att2(item_emb, user_hist, hist_len, user_hist_click)


        # inner_p = self.inner_product(torch.cat((embed_x,user_rep.view(-1,self.item_num_fields,self.embed_dim)),dim=1))
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, inner_p), dim=1)
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, label_rep), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim), theme_hist_rep),dim=1)
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out),score

class MY_MODEL_DEBIAS(Rec):
    def __init__(self, model_config, data_config):
        super(MY_MODEL_DEBIAS, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.label_embedding = FeaturesEmbedding(2, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.num_fields+2*self.item_num_fields+1)*self.embed_dim
        mlp_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.mlp2_dim = self.domain_num_fields*self.embed_dim
        self.mlp2 = MultiLayerPerceptron(self.mlp2_dim,self.hidden_dims,self.dropout)
        self.att = HistAtt2(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        
        self.coAtt = CoAtt2(self.gru_h_dim)
        # self.plelayer = PLELayer(self.expert_num,self.hidden_dims,self.tower_dims,self.task_num,self.embed_dim,mlp_dim,self.dropout)
        
    def get_name(self):
        return 'MY_MODEL_DEBIAS'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist_click = theme_hist[:,:,[-1]]
        theme_hist = theme_hist[:,:,:-1]
        user_hist_click = user_hist[:,:,[-1]]
        user_hist = user_hist[:,:,:-1]

        d_inp = self.embedding(x_domain).view(-1,self.mlp2_dim)
        y_d = self.mlp2(d_inp).squeeze(1)
      
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist_click = self.label_embedding(user_hist_click).view(-1,user_hist_click.shape[1],user_hist_click.shape[2]*self.embed_dim)
        
        theme_hist = self.embedding(theme_hist).view(-1, theme_hist.shape[1], theme_hist.shape[2] * self.embed_dim)
        
        theme_hist_gru_reps, _ = self.gru(theme_hist)
        mask1 = torch.arange(theme_hist_gru_reps.shape[1])[None, :].to(theme_hist_len.device) < theme_hist_len[:, None]
        mask2 = torch.arange(theme_hist_gru_reps.shape[1])[None, :].to(theme_hist_len.device) < (theme_hist_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,theme_hist_gru_reps.shape[1],1),(1,theme_hist_gru_reps.shape[2]))
        theme_gru_reps = torch.sum((torch.mul(theme_hist_gru_reps,mask)),axis= 1)

        x = torch.cat((x_user, x_item,x_domain), dim=1)
        
        embed_x = self.embedding(x)
   
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        
        theme_user_rep, theme_label_rep, score = self.att(item_emb, theme_hist, hist_len, theme_hist_click)
        
        # user_rep, user_label_rep,score = self.att(item_emb,user_hist,hist_len,user_hist_click)
        # user_rep, user_label_rep,score = self.coAtt(item_emb,theme_hist,theme_hist_len,user_hist,hist_len,user_hist_click)
        # inp = torch.cat((embed_x.view(-1,self.input_dim),theme_user_rep,user_rep,user_label_rep), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim),theme_user_rep), dim=1)
        # inp = torch.cat((embed_x.view(-1,self.input_dim),theme_gru_reps,user_rep,user_label_rep), dim=1)
        # results = self.plelayer(inp)
        results = self.mlp(inp)
        y_uid = results.squeeze(1)
        return y_uid*torch.sigmoid(y_d),y_d


class MY_MODEL(Rec):
    def __init__(self, model_config, data_config):
        super(MY_MODEL, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.label_embedding = FeaturesEmbedding(2, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.num_fields+2*self.item_num_fields+1)*self.embed_dim
        # mlp_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt2(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)

        self.coAtt = CoAtt2(self.gru_h_dim)
        self.plelayer = PLELayer(self.expert_num,self.hidden_dims,self.tower_dims,self.task_num,self.embed_dim,mlp_dim,self.dropout)
        
    def get_name(self):
        return 'MY_MODEL'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist_click = theme_hist[:,:,[-1]]
        theme_hist = theme_hist[:,:,:-1]
        user_hist_click = user_hist[:,:,[-1]]
        user_hist = user_hist[:,:,:-1]
      
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist_click = self.label_embedding(user_hist_click).view(-1,user_hist_click.shape[1],user_hist_click.shape[2]*self.embed_dim)
        # theme_hist_click = self.label_embedding(theme_hist_click).view(-1,theme_hist_click.shape[1],theme_hist_click.shape[2]*self.embed_dim)
        
        theme_hist = self.embedding(theme_hist).view(-1, theme_hist.shape[1], theme_hist.shape[2] * self.embed_dim)
        
        theme_hist_gru_reps, _ = self.gru(theme_hist)
        mask1 = torch.arange(theme_hist_gru_reps.shape[1])[None, :].to(theme_hist_len.device) < theme_hist_len[:, None]
        mask2 = torch.arange(theme_hist_gru_reps.shape[1])[None, :].to(theme_hist_len.device) < (theme_hist_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,theme_hist_gru_reps.shape[1],1),(1,theme_hist_gru_reps.shape[2]))
        theme_gru_reps = torch.sum((torch.mul(theme_hist_gru_reps,mask)),axis= 1)

        x = torch.cat((x_user, x_item,x_domain), dim=1)
        
        embed_x = self.embedding(x)
   
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        
        # user_rep, user_label_rep, score = self.att(item_emb, user_hist, hist_len, user_hist_click)

        theme_user_rep, theme_label_rep, score = self.att(item_emb, theme_hist, hist_len, theme_hist_click)
        
        # mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        # mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        # theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        # theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        # theme_user_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        user_rep, user_label_rep,score = self.coAtt(item_emb,theme_hist,theme_hist_len,user_hist,hist_len,user_hist_click)

        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, inner_p), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim),theme_user_rep,user_rep,user_label_rep), dim=1)
        # inp = torch.cat((embed_x.view(-1,self.input_dim),theme_gru_reps,user_rep,user_label_rep), dim=1)
        # inp = torch.cat((embed_x.view(-1,self.input_dim),theme_user_rep), dim=1)
        results = self.mlp(inp)
        # results = self.plelayer(inp)
        out = results.squeeze(1)

        return torch.sigmoid(out),None


class MMOE(Rec):
    def __init__(self, model_config, data_config):
        super(MMOE, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        
        self.input_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.input_dim,self.hidden_dims,self.dropout,output_layer = False) for i in range(self.expert_num)])
        
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(self.hidden_dims[-1],self.tower_dims,self.dropout) for i in range(self.task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.expert_num), torch.nn.Softmax(dim=1)) for i in range(self.task_num)])
        
        
    def get_name(self):
        return 'MMOE'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_dim)
        embed_x = torch.cat((embed_x,theme_hist_rep),dim=1)
        # gate_value = self.gate(embed_x).unsqueeze(1) 
        # fea = torch.cat([self.expert[i](embed_x).unsqueeze(1) for i in range(self.expert_num)],dim=1)
        # print(fea.shape) 1024*3*80
        # print(gate_value.shape) 1024*1*3
        # task_fea = torch.bmm(gate_value,fea).squeeze(1)
        gate_value = [self.gate[i](embed_x).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](embed_x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        
        results = torch.cat([torch.sigmoid(self.tower[i](task_fea[i])) for i in range(self.task_num)],dim=1)
        # print(results.shape)
        # print(results)
        # mask_domain = (F.one_hot(x_domain-self.domain_num,num_classes=3)).squeeze()
        # print(mask_domain.shape)
        score = None
        out = results.squeeze(1)
        # out = torch.sum(mask_domain*results,dim=1)
        # print(out)
        
        # out = self.tower(task_fea).squeeze(1)
        
        return out,score


class SHARED_BOTTOM(Rec):
    def __init__(self, model_config, data_config):
        super(SHARED_BOTTOM, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        
        self.bottom = MultiLayerPerceptron(self.input_dim, self.hidden_dims,self.dropout,output_layer=False)
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(self.hidden_dims[-1],self.tower_dims,self.dropout) for i in range(self.task_num)])
    
    def get_name(self):
        return 'SHARED_BOTTOM'


    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        x = torch.cat((x_user,x_item,x_domain),dim=1)
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_dim)
        embed_x = torch.cat((embed_x,theme_hist_rep),dim=1)
        fea = self.bottom(embed_x)

        results = torch.cat([torch.sigmoid(self.tower[i](fea)) for i in range(self.task_num)],dim=1)
        # mask_domain = (F.one_hot(x_domain-self.domain_num,num_classes=3)).squeeze()
        
        score =None
        
        # result = torch.sum(mask_domain*results,dim=1)
        result = results.squeeze(1)
        return result,None



class STAR(Rec):
    def __init__(self, model_config, data_config):
        super(STAR, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        
        self.input_dim = self.num_fields*self.embed_dim
        self.central_network = MultiLayerPerceptron(self.input_dim,self.hidden_dims,self.dropout,)


    def get_name(self):
        return 'STAR'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_x)

        domain_id = (x_domain[0][0]-self.domain_num)

        gate_value = self.gate(embed_x).unsqueeze(1) 
        fea = torch.cat([self.expert[i](embed_x).unsqueeze(1) for i in range(self.expert_num)],dim=1)
        # print(fea.shape) 1024*3*80
        # print(gate_value.shape) 1024*1*3
        task_fea = torch.bmm(gate_value,fea).squeeze(1)

        out = self.tower(task_fea).squeeze(1)
        score =None
        return torch.sigmoid(out),score


class PLE(Rec):
    """
    A pytorch implementation of PLE Model.
    Reference:
        Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
    """

    def __init__(self, model_config,data_config):
        super(PLE,self).__init__(model_config,data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size,self.embed_dim)
        self.input_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        self.plelayer = PLELayer(self.expert_num,self.hidden_dims,self.tower_dims,self.task_num,self.embed_dim,self.input_dim,self.dropout)
    def get_name(self):
        return 'PLE'
        
    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        x = torch.cat((x_user,x_item,x_domain),dim=1)
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_dim)
        embed_x = torch.cat((embed_x,theme_hist_rep),dim=1)

        # results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        results = self.plelayer(embed_x)
        # mask_domain = (F.one_hot(x_domain-self.domain_num,num_classes=3)).squeeze()
        
        score =None
        
        # result = torch.sum(mask_domain*results,dim=1)
        result = results.squeeze(1)
        
        return torch.sigmoid(result),score




class AESM2(Rec):
    def __init__(self, model_config, data_config):
        super(AESM2, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        
        self.input_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        self.experts = torch.nn.ModuleList([MultiLayerPerceptron(self.input_dim,self.hidden_dims,self.dropout,output_layer = False) for i in range(self.expert_num)])
        
        self.tower = MultiLayerPerceptron(self.hidden_dims[-1],self.tower_dims,self.dropout)
        self.gates = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.domain_size) for i in range(self.expert_num)])
        self.k = 3
        self.null_attention = -2 ** 10
        
    def get_name(self):
        return 'AESM2'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_dim)
        embed_x = torch.cat((embed_x,theme_hist_rep),dim=1)

        gate_value = torch.cat([self.gates[i](embed_x).unsqueeze(0) for i in range(self.expert_num)]) # 6*1024*3
        # print(gate_value.shape)
        gate_value_sm = torch.nn.Softmax(dim=2)(gate_value)
        # gate_value = torch.cat(gate_value,dim=1) # 1024*6*3
        # print(gate_value.shape)
        domain_id = (x_domain[0][0]-self.domain_num)

        p_j = F.one_hot(domain_id,num_classes=self.domain_size)
        q_j = (torch.ones(self.domain_size)/self.domain_size).to(x_domain.device)
        h_p = torch.tensor([-torch.nn.KLDivLoss(reduction = 'batchmean')(p_j,value) for value in gate_value_sm])
        h_q = torch.tensor([-torch.nn.KLDivLoss(reduction = 'batchmean')(q_j,value) for value in gate_value_sm])
        p_topk_indices = torch.topk(h_p,self.k)[1].tolist()
        q_topk_indices = torch.topk(h_q,self.k)[1].tolist()
        chosen_indices = list(set(p_topk_indices+q_topk_indices))
        
        gate_value_d = gate_value[:,:,domain_id].permute(1,0) #1024*6
        gate_value_d[:,chosen_indices] = self.null_attention
        gate_value_d = torch.nn.Softmax(dim=1)(gate_value_d).unsqueeze(1) #1024*1*expert_num
        

        # gate_value = self.gate(embed_x).unsqueeze(1) 
        feas = torch.cat([self.experts[i](embed_x).unsqueeze(1) for i in range(self.expert_num)],dim=1)
        # print(feas.shape) 1024*expert_num*hidden_dims[-1]
        # print(gate_value.shape) 1024*1*expert_num
        task_fea = torch.bmm(gate_value_d,feas).squeeze(1) #1024*hidden_dims[-1]

        out = self.tower(task_fea).squeeze(1)

        score = None
        
        return torch.sigmoid(out),score



class M2M(Rec):
    def __init__(self, model_config, data_config):
        super(M2M, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        
        self.input_dim = (self.num_fields + self.item_num_fields-self.domain_num_fields)*self.embed_dim
        self.domain_bottom = MultiLayerPerceptron(self.domain_num_fields * self.embed_dim,self.hidden_dims,self.dropout,output_layer=False)
        
        self.transformerEncoder = torch.nn.TransformerEncoderLayer(d_model=self.item_num_fields*self.embed_dim, nhead = 8,batch_first = True)
     
        self.experts = torch.nn.ModuleList([MultiLayerPerceptron(self.input_dim,self.hidden_dims,self.dropout,output_layer = False) for i in range(self.expert_num)])
        self.meta_attention = MetaAttention(self.expert_num,self.hidden_dims[-1], self.hidden_dims[-1],self.meta_dims)
        self.tower1 = MetaUnit(self.hidden_dims[-1],self.hidden_dims[-1],self.meta_dims,output_layer=False)
        # self.tower2 = MetaUnit(2*self.hidden_dims[-1],self.hidden_dims[-1],self.meta_dims,output_layer=True)
        self.tower3 = MultiLayerPerceptron(self.hidden_dims[-1]+self.meta_dims[-1]+self.hidden_dims[-1],self.meta_dims,self.dropout,output_layer=True)
    def get_name(self):
        return 'M2M'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        theme_hist = self.transformerEncoder(theme_hist)
        
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        x = torch.cat((x_user,x_item),dim=1)
        embed_x = self.embedding(x).view(-1,(self.num_fields-self.domain_num_fields)*self.embed_dim)
        embed_x = torch.cat((embed_x,theme_hist_rep),dim=1)

        embed_domain = self.embedding(x_domain).view(-1,self.domain_num_fields*self.embed_dim)
        domain_reps = self.domain_bottom(embed_domain)
        feas = [self.experts[i](embed_x) for i in range(self.expert_num)]
        x_reps = self.meta_attention(feas,domain_reps)
        tower1_output = self.tower1(feas[0],domain_reps)
        x_reps = torch.cat([feas[0],tower1_output,x_reps],dim=1)
        # tower1_output = self.tower1(x_reps,domain_reps)
        tower1_output = self.tower3(x_reps)
        # x_reps = torch.cat([x_reps,tower1_output],dim=1)
        # result = self.tower2(x_reps,domain_reps).squeeze(1)
        result = tower1_output.squeeze(1)
        score =None
        return torch.sigmoid(result),score

















class DIN_SESSION(Rec):
    def __init__(self, model_config, data_config):
        super(DIN_SESSION, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.user_num_fields+3*self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        
        self.input_dim = self.num_fields*self.embed_dim
        
    def get_name(self):
        return 'DIN_SESSION'

    def forward(self, x_user, x_item,x_sn, user_hist, hist_len,x_session, session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
        
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        x_session = self.embedding(x_session).view(-1,x_session.shape[1],x_session.shape[2] * self.embed_dim)
        x_session_reps, _ = self.gru(x_session)

        mask1 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < session_len[:, None]
        mask2 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < (session_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,x_session_reps.shape[1],1),(1,x_session_reps.shape[2]))
        x_session_reps = torch.sum((torch.mul(x_session_reps,mask)),axis= 1)


        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        user_rep, score = self.att(item_emb, user_hist, hist_len)

        score = None

        # inner_p = self.inner_product(torch.cat((embed_x,user_rep.view(-1,self.item_num_fields,self.embed_dim)),dim=1))
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, inner_p), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep,x_session_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out),score



class COSMO(Rec):
    def __init__(self, model_config, data_config):
        super(COSMO, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.user_num_fields+3*self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att4item = HistAtt(self.item_num_fields * self.embed_dim)
        self.att4session = HistAtt(self.gru_h_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        self.fc1 = torch.nn.Linear(2*self.item_num_fields*self.embed_dim,self.item_num_fields*self.embed_dim)
        
    def get_name(self):
        return 'COSMO'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
     
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        x_session = self.embedding(x_session).view(-1,x_session.shape[1],x_session.shape[2] * self.embed_dim)
        x_session_reps, _ = self.gru(x_session)

        mask1 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < session_len[:, None]
        mask2 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < (session_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,x_session_reps.shape[1],1),(1,x_session_reps.shape[2]))
        x_session_reps = torch.sum((torch.mul(x_session_reps,mask)),axis= 1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        item_sess_q = self.fc1(torch.cat([item_emb,x_session_reps],dim=1))

        user_rep, item_atten_score = self.att4item(item_sess_q, user_hist, hist_len)
        # user_rep, item_atten_score = self.att4item(item_emb, user_hist, hist_len)
        
        
        # inner_p = self.inner_product(torch.cat((embed_x,user_rep.view(-1,self.item_num_fields,self.embed_dim)),dim=1))
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, inner_p), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep,x_session_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), item_atten_score

class DIEN(Rec):
    def __init__(self, model_config, data_config):
        super(DIEN, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.user_num_fields+2*self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.item_num_fields *self.embed_dim
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        
        
    def get_name(self):
        return 'DIEN'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):

        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]

        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        user_hist_reps, _ = self.gru(user_hist)
        user_rep, atten_score = self.att(item_emb, user_hist_reps, hist_len)

        inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep), dim=1)

        # inner_p = self.inner_product(torch.cat((embed_x,user_rep.view(-1,self.item_num_fields,self.embed_dim)),dim=1))
        # inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, inner_p), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out),atten_score



class UBR(Rec):
    """
    A pytorch implementation of UBR as recall model
    """

    def __init__(self, model_config, data_config):
        super(UBR, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + 253
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt(self.item_num_fields * self.embed_dim)
        self.input_dim = self.num_fields *self.embed_dim
        self.cross_net = CrossNet(self.input_dim, 5)
        self.inner_product = InnerProductNetwork()
        
    def get_name(self):
        return 'UBR'

    def forward(self, x_user, x_item, x_stat, user_hist=None, hist_len=None, ubr_user_hist=None, ubr_hist_len=None):
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
          
        # user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        ubr_user_hist = self.embedding(ubr_user_hist).view(-1, ubr_user_hist.shape[1],ubr_user_hist.shape[2] * self.embed_dim)

        # user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        ubr_user_rep, ubr_atten_score = self.att(item_emb, ubr_user_hist, ubr_hist_len)

        x = torch.cat((x_user, x_item, x_stat), dim=1)
        embed_x = self.embedding(x)
        # cross_out = self.cross_net(embed_x.view(-1,self.input_dim))

        inner_p = self.inner_product(torch.cat((embed_x,ubr_user_rep.view(-1,self.item_num_fields,self.embed_dim)),dim=1))
        #inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep, ubr_user_rep, cross_out), dim=1)
        inp = torch.cat((embed_x.view(-1,self.input_dim), ubr_user_rep, inner_p), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out)


class GRU4REC(Rec):
    def __init__(self, model_config, data_config):
        super(GRU4REC, self).__init__(model_config, data_config)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields+2*self.item_num_fields)*self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.input_dim = self.embed_dim*self.num_fields
     
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
      
    def get_name(self):
        return 'GRU4REC'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
        
     
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        user_hist = self.embedding(user_hist).view(-1,user_hist.shape[1],user_hist.shape[2] * self.embed_dim)
        user_hist_reps, _ = self.gru(user_hist)

        mask1 = torch.arange(user_hist_reps.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        mask2 = torch.arange(user_hist_reps.shape[1])[None, :].to(hist_len.device) < (hist_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,user_hist_reps.shape[1],1),(1,user_hist_reps.shape[2]))
        user_hist_reps = torch.sum((torch.mul(user_hist_reps,mask)),axis= 1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        score = None

        inp = torch.cat((embed_x.view(-1,self.input_dim),user_hist_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), score

class BST(Rec):
    def __init__(self,model_config,data_config):
        super(BST,self).__init__(model_config,data_config)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.position_embedding = torch.nn.Embedding(self.max_hist_len, self.embed_dim)
        self.input_dim = self.embed_dim*self.num_fields
        
        mlp_dim = self.input_dim + self.embed_dim*self.item_num_fields*self.max_hist_len
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        
        self.transformerEncoder = torch.nn.TransformerEncoderLayer(d_model=self.item_num_fields*self.embed_dim, nhead = 8)
     
    def get_name(self):
        return 'BST'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]

        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist_emb = self.embedding(user_hist)

        position = torch.arange(self.max_hist_len).to(x_user.device)
        position_emb = self.position_embedding(position).unsqueeze(dim=1)
        position_emb = torch.tile(position_emb,[user_hist_emb.shape[0],1,user_hist_emb.shape[2],1])
        user_hist_emb += position_emb

        user_hist_emb = user_hist_emb.view(-1,self.max_hist_len,self.item_num_fields*self.embed_dim)
        user_hist_emb = user_hist_emb.permute(1,0,2)   
        user_hist_reps = self.transformerEncoder(user_hist_emb)
        user_hist_reps = user_hist_emb.permute(1,0,2)
        user_hist_reps = user_hist_reps.view(x_user.shape[0],-1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        score = None

        inp = torch.cat((embed_x.view(-1,self.input_dim),user_hist_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), score

class MISS(Rec):
    def __init__(self,model_config,data_config):
        super(MISS,self).__init__(model_config,data_config)

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)

        mlp_dim = 3*self.embed_dim*self.item_num_fields
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)

        
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru_hist = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        self.gru_session = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
         
      
        self.fc1 = torch.nn.Linear(self.gru_h_dim,self.embed_dim)
        self.fc2 = torch.nn.Linear(self.embed_dim,self.n_heads)
        self.fc3 = torch.nn.Linear(2*self.embed_dim*self.item_num_fields, self.embed_dim*self.item_num_fields)

        
    def get_name(self):
        return 'MISS'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]

        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)

        user_hist = self.embedding(user_hist).view(-1,user_hist.shape[1],user_hist.shape[2] * self.embed_dim)
        # user_hist_reps, _ = self.gru_hist(user_hist)
        user_hist_reps = user_hist

        hist_sessions_reps = torch.mean(user_hist_reps.reshape(-1,self.hist_session_num,self.hist_session_length,self.gru_h_dim),dim=2)

        att_score = self.fc2(torch.tanh(self.fc1(hist_sessions_reps)))
        att_score = torch.nn.Softmax(dim=1)(att_score.squeeze())
        latent_user_reps = torch.sum((att_score.unsqueeze(dim=3)*torch.tile(hist_sessions_reps,[1,1,self.n_heads]).reshape(-1,self.hist_session_num,self.n_heads,self.gru_h_dim)),dim=1)
        
        x_session = self.embedding(x_session).view(-1,x_session.shape[1],x_session.shape[2] * self.embed_dim)
        # x_session_reps, _ = self.gru_session(x_session)
        x_session_reps = torch.mean(x_session,dim=1)

        user_att_score = torch.nn.Softmax(dim=1)(torch.bmm(latent_user_reps,x_session_reps.unsqueeze(dim=2)).squeeze())
        user_emb = torch.sum(user_att_score.unsqueeze(dim=2)*latent_user_reps,dim = 1)

        score = None

        z_u = self.fc3(torch.cat((user_emb,x_session_reps),dim=1))

        out = torch.sum(z_u*item_emb,dim=1)

        return torch.sigmoid(out), score
        

class COSMO2(Rec):
    def __init__(self, model_config, data_config):
        super(COSMO2, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.user_num_fields+3*self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att4item = HistAtt(self.item_num_fields * self.embed_dim)
        self.att4session = HistAtt(self.gru_h_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)

        self.coAtt = CoAtt(self.gru_h_dim)
        
    def get_name(self):
        return 'COSMO2'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
     
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        x_session = self.embedding(x_session).view(-1,x_session.shape[1],x_session.shape[2] * self.embed_dim)
        x_session_reps, _ = self.gru(x_session)

        mask1 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < session_len[:, None]
        mask2 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < (session_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,x_session_reps.shape[1],1),(1,x_session_reps.shape[2]))
        session_reps = torch.sum((torch.mul(x_session_reps,mask)),axis= 1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, att_score = self.coAtt(item_emb,x_session_reps,session_len,user_hist,hist_len)
        
        inp = torch.cat((embed_x.view(-1,self.input_dim), user_rep,session_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), att_score


class NCF(Rec):
    def __init__(self, model_config, data_config):
        super(NCF, self).__init__(model_config, data_config)
        self.input_dim = self.num_fields*self.embed_dim
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.num_fields+self.item_num_fields)*self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        
    def get_name(self):
        return 'NCF'

    def forward(self, x_user, x_item,x_domain,theme_hist, theme_hist_len, user_hist, hist_len):
        theme_hist = theme_hist[:,:,:-1]
        x = torch.cat((x_user, x_item, x_domain), dim=1)
        mask_hist = (theme_hist[:,:,-1] !=0).to(theme_hist.device) # bs* length
        theme_hist =self.embedding(theme_hist).view(-1,theme_hist.shape[1],self.item_num_fields*self.embed_dim)
        mask_hist = torch.tile(mask_hist.unsqueeze(2),[1,1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.tile(theme_hist_len.unsqueeze(1),[1,theme_hist.shape[2]])
        theme_hist_len_tiled = torch.where(theme_hist_len_tiled>0,theme_hist_len_tiled,theme_hist_len_tiled+1)
        theme_hist_rep = torch.sum(mask_hist*theme_hist,dim=1)/theme_hist_len_tiled #bs * embed_dim  mean pooling
        
        embed_x = self.embedding(x).view(-1,self.num_fields*self.embed_dim)
        inp = torch.cat((embed_x,theme_hist_rep),dim=1)
        
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), None


class NCF_DEBIAS(Rec):
    def __init__(self, model_config, data_config):
        super(NCF_DEBIAS, self).__init__(model_config, data_config)
        self.input_dim = self.num_fields*self.embed_dim
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = self.num_fields*self.embed_dim
        self.mlp2_dim = self.domain_num_fields*self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.mlp2 = MultiLayerPerceptron(self.mlp2_dim,self.hidden_dims,self.dropout)
    def get_name(self):
        return 'NCF_DEBIAS'

    def forward(self, x_user, x_item, x_domain,user_hist, hist_len):
        x = torch.cat((x_user, x_item,x_domain), dim=1)
        
        embed_x = self.embedding(x)

        inp =embed_x.view(-1,self.input_dim)
        d_inp = self.embedding(x_domain).view(-1,self.mlp2_dim)
        y_uid = self.mlp(inp).squeeze(1)
        y_d = self.mlp2(d_inp).squeeze(1)


        return y_uid*torch.sigmoid(y_d), y_d


class Pinet(Rec):
    def __init__(self, model_config, data_config):
        super(Pinet, self).__init__(model_config, data_config)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields+4*self.item_num_fields)*self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.input_dim = self.embed_dim*self.num_fields
     
        self.gru_h_dim = self.embed_dim*self.item_num_fields

        self.gru1 = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        self.gru2 = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
        self.gru3 = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)
      
    def get_name(self):
        return 'Pinet'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
        
     
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        user_hist = self.embedding(user_hist).view(-1,user_hist.shape[1],user_hist.shape[2] * self.embed_dim)
        user_hist_reps1, _ = self.gru1(user_hist)
        user_hist_reps2, _ = self.gru2(user_hist)
        user_hist_reps3, _ = self.gru3(user_hist)

        mask1 = torch.arange(user_hist_reps1.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        mask2 = torch.arange(user_hist_reps1.shape[1])[None, :].to(hist_len.device) < (hist_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,user_hist_reps1.shape[1],1),(1,user_hist_reps1.shape[2]))
        user_hist_reps1 = torch.sum((torch.mul(user_hist_reps1,mask)),axis= 1)
        user_hist_reps2 = torch.sum((torch.mul(user_hist_reps2,mask)),axis= 1)
        user_hist_reps3 = torch.sum((torch.mul(user_hist_reps3,mask)),axis= 1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        score = None

        inp = torch.cat((embed_x.view(-1,self.input_dim),user_hist_reps1,user_hist_reps2,user_hist_reps3), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), score
        

class ISN(Rec):
    def __init__(self, model_config, data_config):
        super(ISN, self).__init__(model_config, data_config)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields+2*self.item_num_fields)*self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.input_dim = self.embed_dim*self.num_fields
        self.fc1 = torch.nn.Linear(self.max_hist_len,3)
        atten_input_dim = 2*self.embed_dim*self.item_num_fields
        layers = list()
        for hidden_dim in [80]:
            layers.append(torch.nn.Linear(atten_input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            atten_input_dim = hidden_dim
        layers.append(torch.nn.Linear(atten_input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)

    def get_name(self):
        return 'ISN'

    def forward(self, x_user, x_item, x_sn,user_hist, hist_len,x_session,session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
        
     
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        user_hist = self.embedding(user_hist).view(-1,user_hist.shape[1],user_hist.shape[2] * self.embed_dim)

        latent_users = self.fc1(user_hist.permute(0,2,1))

        atten_input = torch.cat([torch.tile(item_emb.unsqueeze(dim=2),[1,1,3]),latent_users],dim= 1).permute(0,2,1)
        atten_output = self.atten_net(atten_input).squeeze()
        atten_score = torch.nn.Softmax(dim=1)(atten_output)

        hist_reps = torch.sum(latent_users.permute(0,2,1)* atten_score.unsqueeze(dim=2),dim=1)

        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        score = None

        inp = torch.cat((embed_x.view(-1,self.input_dim),hist_reps), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out), score

class DIN_BPR(Rec):
    def __init__(self, model_config, data_config):
        super(DIN_BPR, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        inter_fields = self.item_num_fields+self.num_fields
        mlp_dim = (self.user_num_fields+3*self.item_num_fields)*self.embed_dim
        # mlp_dim = (self.item_num_fields+self.num_fields) * self.embed_dim + int(inter_fields * (inter_fields - 1) / 2)
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout)
        self.att = HistAtt(self.item_num_fields * self.embed_dim)
        self.inner_product = InnerProductNetwork()
        self.input_dim = self.num_fields*self.embed_dim
        self.gru_h_dim = self.embed_dim*self.item_num_fields
        self.gru = torch.nn.GRU(self.item_num_fields * self.embed_dim, self.gru_h_dim, batch_first = True)

        
    def get_name(self):
        return 'DIN_BPR'

    def forward(self, x_user, x_item,x_sn, user_hist, hist_len,x_session, session_len, ubr_user_hist=None, ubr_hist_len=None):
        
        user_hist = user_hist[:,:,:-1]
        x_session = x_session[:,:,:-1]
        
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        
        x_session = self.embedding(x_session).view(-1,x_session.shape[1],x_session.shape[2] * self.embed_dim)
        x_session_reps, _ = self.gru(x_session)

        mask1 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < session_len[:, None]
        mask2 = torch.arange(x_session_reps.shape[1])[None, :].to(session_len.device) < (session_len-1)[:, None]
        
        mask = torch.logical_xor(mask1,mask2)
        mask = torch.tile(mask.reshape(-1,x_session_reps.shape[1],1),(1,x_session_reps.shape[2]))
        x_session_reps = torch.sum((torch.mul(x_session_reps,mask)),axis= 1)

        
        x = torch.cat((x_user, x_item), dim=1)
        embed_x = self.embedding(x)

        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        user_rep, score = self.att(item_emb, user_hist, hist_len)


        inp = torch.cat((embed_x.view(-1,self.input_dim),x_session_reps, user_rep), dim=1)
        out = self.mlp(inp).squeeze(1)
        return torch.sigmoid(out),score



