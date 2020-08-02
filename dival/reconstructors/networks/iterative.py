import torch
import torch.nn as nn


class IterativeBlock(nn.Module):
    def __init__(self, n_in=3, n_out=1, n_memory=5, n_layer=3, internal_ch=32,
                 kernel_size=3, batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(IterativeBlock, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        modules = []
        if batch_norm:
            modules.append(nn.BatchNorm2d(n_in + n_memory))
        for i in range(n_layer-1):
            input_ch = (n_in + n_memory) if i == 0 else internal_ch
            modules.append(nn.Conv2d(input_ch, internal_ch,
                                     kernel_size=kernel_size, padding=padding))
            if batch_norm:
                modules.append(nn.BatchNorm2d(internal_ch))
            if prelu:
                modules.append(nn.PReLU(internal_ch, init=0.0))
            else:
                modules.append(nn.LeakyReLU(lrelu_coeff, inplace=True))
        modules.append(nn.Conv2d(internal_ch, n_out + n_memory,
                                 kernel_size=kernel_size, padding=padding))
        self.block = nn.Sequential(*modules)
        self.relu = nn.LeakyReLU(lrelu_coeff, inplace=True)  # remove?

    def forward(self, x):
        upd = self.block(x)
        return upd


class IterativeNet(nn.Module):
    def __init__(self, n_iter, n_memory, op, op_adj, op_init, op_reg,
                 use_sigmoid=False, n_layer=4, internal_ch=32,
                 kernel_size=3, batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(IterativeNet, self).__init__()
        self.n_iter = n_iter
        self.n_memory = n_memory
        self.op = op
        self.op_adj = op_adj
        self.op_init = op_init
        self.op_reg = op_reg
        self.use_sigmoid = use_sigmoid

        self.blocks = nn.ModuleList()
        for it in range(n_iter):
            self.blocks.append(IterativeBlock(
                n_in=3, n_out=1, n_memory=self.n_memory, n_layer=n_layer,
                internal_ch=internal_ch, kernel_size=kernel_size,
                batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))

    def forward(self, y, it=-1):
        if self.op_init is not None:
            x = self.op_init(y)
        else:
            x = torch.zeros(y.shape[0], 1, *self.op.operator.domain.shape,
                            device=y.device)
        s = torch.zeros(x.shape[0], self.n_memory, *x.shape[2:],
                        device=y.device)
        n_iter = self.n_iter if it == -1 else min(self.n_iter, it)
        for i in range(n_iter):
            grad_x = self.op_adj(self.op(x) - y)
            grad_reg = self.op_reg(x)
            update = self.blocks[i](torch.cat([x, grad_x, grad_reg, s], dim=1))
            x += update[:, :1, ...]
            s = torch.relu(update[:, 1:, ...])
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


class PrimalDualNet(nn.Module):
    def __init__(self, n_iter, op, op_adj, op_init=None, n_primal=5, n_dual=5,
                 use_sigmoid=False, n_layer=4, internal_ch=32, kernel_size=3,
                 batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(PrimalDualNet, self).__init__()
        self.n_iter = n_iter
        self.op = op
        self.op_adj = op_adj
        self.op_init = op_init
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.use_sigmoid = use_sigmoid

        self.primal_blocks = nn.ModuleList()
        self.dual_blocks = nn.ModuleList()
        for it in range(n_iter):
            self.dual_blocks.append(IterativeBlock(
                n_in=3, n_out=1, n_memory=self.n_dual-1, n_layer=n_layer,
                internal_ch=internal_ch, kernel_size=kernel_size,
                batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))
            self.primal_blocks.append(IterativeBlock(
                n_in=2, n_out=1, n_memory=self.n_primal-1, n_layer=n_layer,
                internal_ch=internal_ch, kernel_size=kernel_size,
                batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))

    def forward(self, y):
        primal_cur = torch.zeros(y.shape[0], self.n_primal,
                                 *self.op.operator.domain.shape,
                                 device=y.device)
        if self.op_init is not None:
            primal_cur[:] = self.op_init(y)  # broadcast across dim=1
        dual_cur = torch.zeros(y.shape[0], self.n_dual,
                               *self.op_adj.operator.domain.shape,
                               device=y.device)
        for i in range(self.n_iter):
            primal_evalop = self.op(primal_cur[:, 1:2, ...])
            dual_update = torch.cat([dual_cur, primal_evalop, y], dim=1)
            dual_update = self.dual_blocks[i](dual_update)
            dual_cur = dual_cur + dual_update
            # NB: currently only linear op supported
            #     for non-linear op: [d/dx self.op(primal_cur[0:1, ...])]*
            dual_evalop = self.op_adj(dual_cur[:, 0:1, ...])
            primal_update = torch.cat([primal_cur, dual_evalop], dim=1)
            primal_update = self.primal_blocks[i](primal_update)
            primal_cur = primal_cur + primal_update
        x = primal_cur[:, 0:1, ...]
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x
