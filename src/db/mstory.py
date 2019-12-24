import torch
import os
import sqlite3
from contextlib import closing
import textwrap
import json
import pandas as pd
import numpy as np
import datetime

try:
    import matplotlib.pylab as plt
except ImportError:
    print('Unable import matplotlib')


def create_table(dbfile):
    with closing(sqlite3.connect(dbfile)) as con:
        cursor = con.cursor()
        # create experiment table
        sql = textwrap.dedent('''\
            create table experiment (
            exid integer primary key autoincrement,
            time timestamp,
            name text,
            arch text,
            path text
            )
            ''')
        print(sql)
        cursor.execute(sql)

        # create history table
        sql = textwrap.dedent('''\
            create table history (
            hid integer primary key autoincrement,
            time timestamp,
            exid integer,
            epoch integer,
            iter integer,
            mode text,
            batch_size integer,
            lr real,
            loss real,
            metrics real
            )
            ''')
        print(sql)
        cursor.execute(sql)

        # create gradient table for observating gradient
        sql = textwrap.dedent('''\
            create table grad (
            gid integer primary key autoincrement,
            exid integer,
            epoch integer,
            iter integer,
            layer_grad blob
            )
            ''')
        print(sql)
        cursor.execute(sql)

        # create loss change allocation(LCA) for observating learn of layers
        sql = textwrap.dedent('''\
            create table lca (
            lid integer primary key autoincrement,
            exid integer,
            epoch integer,
            iter integer,
            layer_lca blob
            )
            ''')
        print(sql)
        cursor.execute(sql)

        con.commit()


class ModelDB:
    """

    Notes
    -----
    The database has 2 tables.
    - experiment table: the table which records model name, architecture and model path.
    - history table: the table which records loss, metrics, lr, batch size, etc. for each iteration.
    - gradient table: the table which records gradients of layers for each iteration.
    - lca table: the table which records loss change allocation of layers for each iteration.
    """
    time_setting = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

    def __init__(self, dbfile):
        """
        Initialize instance.

        Paramters
        ---------
        dbfile: str
            file path of sqlite
        model: torch.nn.Module
        """
        if not os.path.exists(dbfile):
            raise ValueError('DB file not found: %s' % dbfile)

        self.dbfile = dbfile
        # experiment id: primary key of experiment table
        self.ex_id = 'no_exid'

    def rec_model(self, model, model_path):
        """
        write model information to experiment table.

        Paramters
        ---------
        model: torch.nn.Module
        """
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()

            sql = 'insert into experiment(time, name, arch, path) values(?, ?, ?, ?)'
            model_name = model.__class__.__name__
            architecture = str(model)
            # self.model_id = hashlib.md5(architecture.encode()).hexdigest()
            now = datetime.datetime.now()
            data = (now, model_name, architecture, model_path)

            cursor.execute(sql, data)
            self.ex_id = cursor.lastrowid

            con.commit()

    def rec_history(self, epoch, iter, mode, batch_size, lr, loss_val, metrics_val):
        """
        Record loss and metrics for a iteration.
        """
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            sql = 'insert into history(time, exid, epoch, iter, mode, batch_size, lr, loss, metrics) values(?, ?, ?, ?, ?, ?, ?, ?, ?)'
            now = datetime.datetime.now()
            data = (now, self.ex_id, epoch, iter, mode,
                    batch_size, lr, loss_val, metrics_val)
            cursor.execute(sql, data)
            con.commit()

    def rec_grad(self, model, epoch, iter):
        """
        Record gradients for a iteration.
        """
        layer_grads = calc_grad(model)
        layer_grads_str = json.dumps(layer_grads)

        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()

            sql = 'insert into grad(exid, epoch, iter, layer_grad) values(?, ?, ?, ?)'
            data = (self.ex_id, epoch, iter, (layer_grads_str))

            cursor.execute(sql, data)
            con.commit()

    def rec_lca(self, lca_dict, epoch, iter):
        """
        Record LCA for a iteration.
        """
        layer_lca = {
            'layers': list(lca_dict.keys()),
            'sum_lca': [lca_dict[k].sum().item() for k in lca_dict.keys()]
        }
        layer_lca_str = json.dumps(layer_lca)

        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()

            sql = 'insert into lca(exid, epoch, iter, layer_lca) values(?, ?, ?, ?)'
            data = (self.ex_id, epoch, iter, (layer_lca_str))

            cursor.execute(sql, data)
            con.commit()

    def table_experiment(self):
        return ExperimentTable(self.dbfile)

    def table_history(self):
        return HistoryTable(self.dbfile)

    def table_lca(self):
        return LcaTable(self.dbfile)

    def get_grad(self, exid, epoch, iter):
        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()
            sql = 'select layer_grad from grad where exid = ? and epoch = ? and iter = ?'
            data = (exid, epoch, iter)
            cursor.execute(sql, data)

            records = cursor.fetchall()
            if len(records) == 0:
                raise ValueError('No records that meet the conditions')
            else:
                layer_grad_str, = records[0]
                layer_grad = json.loads(layer_grad_str)

        return layer_grad

    def get_lca(self, exid, epoch, iter):
        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()
            sql = 'select layer_lca from lca where exid = ? and epoch = ? and iter = ?'
            data = (exid, epoch, iter)
            cursor.execute(sql, data)

            records = cursor.fetchall()
            if len(records) == 0:
                raise ValueError('No records that meet the conditions')
            else:
                layer_lca_str, = records[0]
                layer_lca = json.loads(layer_lca_str)

        return layer_lca


class TableBinder:
    """
    This class binds sqlite db and provide interfaces like pandas.DataFrame.

    Usage
    -----
    Subclass of this class must be implemented `select` method.
    """

    def __init__(self):
        pass

    def __getitem__(self, i):
        if type(i) == int:
            return self.select(i, 1)
        elif type(i) == slice:
            if i.step is None:
                offset = i.start
                if i.stop is None:
                    # case slice = [offset, None]
                    limit = None
                else:
                    limit = i.stop - i.start
                return self.select(offset, limit)
            else:
                raise IndexError('Unsupport select way')
        else:
            raise IndexError('Unsupport select way')

    def show(self):
        return self.select(-1, -1)

    def head(self, n=5):
        return self.__getitem__(slice(0, n))

    def tail(self, n=5):
        # TODO: implement
        return self.__getitem__(slice(-n, None))

    def select(self, offset, limit):
        raise NotImplementedError


class ExperimentTable(TableBinder):

    time_setting = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

    def __init__(self, dbfile):
        self.dbfile = dbfile

    def select(self, offset, limit):
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            if offset < 0 or limit < 0:
                # select all
                sql = 'select exid, time, name, path from experiment'
                cursor.execute(sql)
            else:
                sql = 'select exid, time, name, path from experiment '\
                    'limit ? offset ?'
                data = (limit, offset)
                cursor.execute(sql, data)

            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=[
                              'exid', 'time', 'name', 'path'])
            return df


class HistoryTable(TableBinder):

    time_setting = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

    def __init__(self, dbfile):
        self.dbfile = dbfile

    def select(self, offset, limit):
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            if offset < 0 or limit < 0:
                # select all
                sql = 'select time, exid, epoch, iter, mode, batch_size, lr, loss, metrics from history'
                cursor.execute(sql)
            else:
                sql = 'select time, exid, epoch, iter, mode, batch_size, lr, loss, metrics from history '\
                    'limit ? offset ?'
                data = (limit, offset)
                cursor.execute(sql, data)

            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=[
                              'time', 'exid', 'epoch', 'iter', 'mode', 'batch_size', 'lr', 'loss', 'metrics'])
            return df


class LcaTable(TableBinder):
    def __init__(self, dbfile):
        self.dbfile = dbfile

    def select(self, offset, limit):
        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()
            if offset < 0 or limit < 0:
                # select all
                sql = 'select lid, exid, epoch, iter, layer_lca from lca'
                cursor.execute(sql)
            else:
                sql = 'select lid, exid, epoch, iter, layer_lca from lca '\
                    'limit ? offset ?'
                data = (limit, offset)
                cursor.execute(sql, data)

            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=[
                              'lid', 'exid', 'epoch', 'iter', 'layer_lca'])

            return df


class GradPlot:
    @classmethod
    def plot_grad(cls, layer_grad, figsize=(16, 6)):
        max_grad = layer_grad['max_abs_grads']
        avg_grad = layer_grad['avg_abs_grads']
        layers = layer_grad['layers']

        # fig, ax = plt.subplots(1, 1)
        plt.figure(figsize=figsize)
        plt.bar(np.arange(len(max_grad)), max_grad, alpha=0.1, lw=1, color='c')
        plt.bar(np.arange(len(max_grad)), avg_grad, alpha=0.1, lw=1, color='b')
        plt.xticks(range(0, len(layers), 1), layers, rotation='vertical')
        plt.xlim(left=-1, right=len(layers))
        plt.xlabel("Layers")
        plt.ylabel("Gradient Magnitude")
        plt.yscale('log')
        plt.title("Gradient flow")
        plt.grid(True)

    @classmethod
    def plot_lca(cls, layer_lca, figsize=(16, 6)):
        """
        Plot LCA

        Parameters
        ----------
        layer_lca: dict
            layer_lca has 'layers' and 'sum_lca' keys.
            - The value of 'layers' is list of layer name.
            - The value of 'sum_lca' is list of sum of LCA for each layer.
        """
        layers = layer_lca['layers']
        sum_lca = layer_lca['sum_lca']

        print(sum_lca)

        plt.figure(figsize=figsize)
        plt.bar(x=np.arange(len(layers)), height=sum_lca,
                alpha=0.5, lw=1, color='c')
        plt.xticks(range(0, len(layers), 1), layers, rotation='vertical')
        plt.xlim(left=-1, right=len(layers))
        plt.xlabel("Layers")
        plt.ylabel("Loss Change Allocation")
        plt.title("Gradient flow")
        plt.grid(True)


def calc_grad(model):
    layers = []
    avg_abs_grads = []
    max_abs_grads = []
    for n, p in model.named_parameters():
        if (p.grad is not None) and ('bias' not in n):
            layers.append(n)
            avg_abs_grads.append(p.grad.abs().mean().item())
            max_abs_grads.append(p.grad.abs().max().item())

    layer_grads = {
        'layers': layers,
        'avg_abs_grads': avg_abs_grads,
        'max_abs_grads': max_abs_grads
    }
    return layer_grads


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    db_path = 'sample.db'
    # create_table('sample.db')
    db = ModelDB(db_path)

    net = MyNet()
    loss = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(32, 10)
    label = torch.bernoulli(torch.empty(32, 1).uniform_(0, 1))

    net.train()
    logit = net(x)
    loss = loss(logit, label)
    loss.backward()

    db.rec_model(net, './model/data224/mynet/001/best_model.pth')
    print('exid: %d' % db.ex_id)

    db.rec_grad(net, 1, 0)

    print(db.get_grad(3, 1, 0))
    df_ex = db.table_experiment()
    print(df_ex.head())
    print(df_ex[1:3])
