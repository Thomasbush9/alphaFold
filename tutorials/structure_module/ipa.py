
import torch
import math
from torch import nn

from geometry.geometry import invert_4x4_transform, precalculate_rigid_transforms, warp_3d_point

class InvariantPointAttention(nn.Module):
    """
    Implements invariant point attention, according to Algorithm 22.
    """

    def __init__(self, c_s, c_z, n_query_points=4, n_point_values=8, N_head=12, c=16):
        """
        Initializes the invariant point attention module.

        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_query_points (int, optional): Number of query points for point attention.
                Used for the embedding of q_points and k_points. Defaults to 4.
            n_point_values (int, optional): Number of value points for point attention.
                Used for the embedding of v_points. Defaults to 8.
            n_head (int, optional): Number of heads for multi-head attention. Defaults to 12.
            c (int, optional): Embedding dimension for each individual head. Defaults to 16.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.N_head = N_head
        self.c = c


        ##########################################################################
        # TODO: Initialize the layers linear_q, linear_k, linear_v,              #
        #   linear_q_points, linear_k_points, linear_v_points, linear_b, and     #
        #   linear_out. The embeddings for q, k and v are similar to             #
        #   MultiHeadAttention, except that they use bias (this clashes with the #
        #   supplement, but follows the official implementation).                #
        #   The point embeddings need to create three values per head and point. #
        #   They also use bias.                                                  #
        #   The embedding for the bias computes one bias value per head.         #
        #   For the input dimension of linear_out, count the channels of the     #
        #   various outputs in line 11 from the algorithm. If you have trouble   #
        #   with this, you can look below at the output description of           #
        #   `compute_outputs`. The output dimension of linear_out is c_s.        #
        #                                                                        #
        #   For the weight per head, gamma, initialize head_weights to a         #
        #   zero-tensor wrapped in nn.Parameter. Also, initialize nn.Softplus    #
        #   for the computation of gamma.                                        #
        ##########################################################################

        # Replace "pass" statement with your code
        self.linear_q = nn.Linear(c_s, self.c*self.N_head)
        self.linear_k = nn.Linear(c_s, self.c*self.N_head)
        self.linear_v = nn.Linear(c_s, self.c*self.N_head)

        self.linear_q_points = nn.Linear(c_s, 3*self.n_query_points*self.N_head)
        self.linear_k_points = nn.Linear(c_s, 3*self.n_query_points*self.N_head)
        self.linear_v_points = nn.Linear(c_s, self.n_point_values*3*self.N_head)
        self.linear_b = nn.Linear(self.c_z, self.N_head)
        self.linear_out = nn.Linear(N_head*c_z+N_head*c+N_head*4*n_point_values, c_s)

        self.head_weights = nn.Parameter(torch.zeros((N_head,)))
        self.softplus = nn.Softplus()
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, s):
        """
        Creates the standard attention embeddings q, k, and v, as well as the point
        embeddings qp, kp, and vp, for invariant point attention.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            tuple: A tuple consisting of the following embeddings:
                q: Tensor of shape (*, N_head, N_res, c)
                k: Tensor of shape (*, N_head, N_res, c)
                v: Tensor of shape (*, N_head, N_res, c)
                qp: Tensor of shape (*, N_head, N_query_poitns, N_res, 3)
                kp: Tensor of shape (*, N_head, N_query_points, N_res, 3)
                vp: Tensor of shape (*, N_head, N_point_values, N_res, 3)
        """
        c = self.c
        n_head = self.N_head
        n_qp = self.n_query_points
        n_pv = self.n_point_values

        embeddings = None

        ##########################################################################
        # TODO: Implement the embedding preparation in the following steps:      #
        #   - Pass s through all of the embedding layers.                        #
        #   - Reshape the feature dimension of the embeddings so that q, k and v #
        #     have shape (*, N_head, c), qp and kp have shape                    #
        #     (*, 3, N_head, n_qp) and vp has shape (*, 3, N_head, n_pv).        #
        #   - Move the dimensions to match the shapes in the method description. #
        ##########################################################################

        # Replace "pass" statement with your code
        # generate embds
        q = self.linear_q(s)
        k = self.linear_k(s)
        v = self.linear_v(s)
        #reshape q, v, k
        batch_dim = q.shape[:-1]
        new_shape = batch_dim + (n_head, c,)
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)
        q = q.movedim(-3, -2)
        k = k.movedim(-3, -2)
        v = v.movedim(-3, -2)

        qp = self.linear_q_points(s)
        kp = self.linear_k_points(s)
        vp = self.linear_v_points(s)
        new_shape_q = batch_dim + (3, n_head, n_qp)
        new_shape_v = batch_dim + (3, n_head, n_pv)
        qp = qp.reshape(new_shape_q)
        kp = kp.reshape(new_shape_q)
        vp = vp.reshape(new_shape_v)
        #move nres
        p_src = (-4, -3)
        p_dst = (-2, -1)
        qp = qp.movedim(p_src, p_dst)
        kp = kp.movedim(p_src, p_dst)
        vp = vp.movedim(p_src, p_dst)
        embeddings = (q, k, v, qp, kp, vp)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return embeddings

    def compute_attention_scores(self, q, k, qp, kp, z, T):
        """
        Computes the attention scores for invariant point attention,
        according to line 7 from Algorithm 22.

        Args:
            q (torch.tensor): Query embeddings of shape (*, N_head, N_res, c).
            k (torch.tensor): Key embeddings of shape (*, N_head, N_res, c).
            qp (torch.tensor): Query point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            kp (torch.tensor): Key point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Attention scores of shape (*, N_head, N_res, N_res).
        """

        att_scores = None

        ##########################################################################
        # TODO: Implement the method in the following steps:                     #
        #   - Compute wc, wl and gamma.                                          #
        #   - Reshape gamma (formerly shape (N_head,) so that it's broadcastable #
        #     against the attention scores.                                      #
        #   - Scale q and compute the bias. Move the dimension of the bias so    #
        #     that it matches the attention scores.                              #
        #   - Compute the qk term. You can use torch.einsum for this.            #
        #   - Reshape the transforms so that they can be used for batched        #
        #     matrix multiplication against the query and key points.            #
        #   - Use warp_3d_point to warp the query and key points through T.      #
        #   - Compute the query points / key points term.                        #
        #   - Compute the full attention scores.                                 #
        ##########################################################################

        # Replace "pass" statement with your code
        wc = math.sqrt(2 / (9* self.n_query_points))
        wl = math.sqrt(1/3)
        gamma = self.softplus(self.head_weights).view((-1, 1, 1))
        q = (1/math.sqrt(self.c)) * q
        bias = self.linear_b(z).movedim(-1, -3)
        qk = torch.einsum('...ic, ...jc->...ij', q, k)
        T_bc_qkv = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        transformed_qp = warp_3d_point(T_bc_qkv, qp).unsqueeze(-2)
        transformed_kp = warp_3d_point(T_bc_qkv, kp).unsqueeze(-3)
        sq_dist = torch.sum((transformed_qp - transformed_kp)**2, dim=-1)
        qpkp_term = gamma * wc / 2 * torch.sum(sq_dist, dim=-3)

        att_scores = torch.softmax(wl * (qk + bias - qpkp_term), dim=-1)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return att_scores

    def compute_outputs(self, att_scores, z, v, vp, T):
        """
        Computes the different output vectors for the IPA attention mechanism:
        The pair output, the standard attention output, and the point attention output,
        as well as the norm of the point attention output.

        Args:
            att_scores (torch.tensor): Attention scores of shape (*, N_head, N_res, N_res).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            v (torch.tensor): Value vectors of shape (*, N_head, N_res, c).
            vp (torch.tensor): Value points of shape (*, N_head, N_point_values, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            tuple: A tuple consisting of the following outputs:
                - output from the value vectors of shape (*, N_res, N_head*c).
                - output from the value points of shape (*, N_res, N_head*3*N_point_values).
                - norm of the output vectors from the value points of shape (*, N_res, N_head*N_point_values)
                - output from the pair representation of shape (*, N_res, N_head*c_z).
        """

        v_out, vp_out, vp_out_norm, pairwise_out = None, None, None, None

        ##########################################################################
        # TODO: Compute the different attention outputs in the following steps:  #
        #   - Compute the pairwise output, move the dimension so that they       #
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Compute the value vector output, move the dimensions so that they  #
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Reshape the transforms so that they can be used for batched        #
        #     matrix multiplication against the value points.                    #
        #   - Warp the value points, compute the point attention values, compute #
        #     the inverse of the transforms with invert_4x4_transform            #
        #     and warp the value points back through them.                       #
        #   - Transpose the axes of the value points from ...hpic to ...ichp     #
        #     (the letters mean N_head, point_values, N_res, c). You can use     #
        #     torch.einsum for this.                                             #
        #   - Compute the vector norms of the point values.                      #
        #   - Flatten the trailing channel, N_head and N_point_value dims for    #
        #     the value points and their norm.                                   #
        ##########################################################################

        # Replace "pass" statement with your code
        # compute pairwise output:
        pairwise_out = torch.einsum('...ijc, ...hij->...hic', z, att_scores).movedim(-3, -2)
        pairwise_out = torch.flatten(pairwise_out, -2, -1)


        v_out = torch.einsum('...hij,...hjc->...hic', att_scores, v)
        v_out = v_out.movedim(-3, -2).flatten(start_dim=-2)
        T_bc_qkv = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        vt = warp_3d_point(T_bc_qkv, vp)
        v_att = torch.einsum('...hij, ...hkjp->...hkip', att_scores, vt)

        T_inv = invert_4x4_transform(T_bc_qkv)
        v_transformed = warp_3d_point(T_inv, v_att)
        v_transformed = torch.einsum('...hpic->...ichp', v_transformed)
        norm = torch.linalg.vector_norm(v_transformed,dim=-3, keepdim=True)
        vp_out_norm = torch.flatten(norm, start_dim=-3)
        vp_out= torch.flatten(v_transformed, start_dim=-3)


        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return v_out, vp_out, vp_out_norm, pairwise_out



    def forward(self, s, z, T):
        """
        Implements the forward pass for InvariantPointAttention, as specified in Algorithm 22.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, c_s).
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass by combining all the methods above.   #
        ##########################################################################

        # Replace "pass" statement with your code
        q, k, v, qp, kp, vp = self.prepare_qkv(s)
        att_scores = self.compute_attention_scores(q, k, qp, kp, z, T)
        v_out, vp_out, vp_out_norm, pairwise_out = self.compute_outputs(att_scores, z, v, vp, T)
        out = torch.cat((v_out, vp_out, vp_out_norm, pairwise_out), dim=-1)
        out = self.linear_out(out)


        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

