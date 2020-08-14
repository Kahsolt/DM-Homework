#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/06/07 

from pdb import set_trace
from os import cpu_count
from sys import argv
from re import compile as REGEX
from time import time, sleep
from os import path
from pprint import pprint as pp
from io import BytesIO
import gzip, pickle
from zipfile import ZipFile, ZIP_LZMA
from threading import Thread, RLock, Event
from queue import Queue, Empty

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from tkinter.scrolledtext import ScrolledText

from pycparser.c_parser import CParser
from pycparser.c_ast import *
from zss import simple_distance, Node      # shadows 'c_ast.Node'

import sqlalchemy as sql
from sqlalchemy.sql import or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from collections import Counter, defaultdict
from enum import Enum
from random import choice, sample, randrange, random
import numpy as np
import pandas as pd
import joblib

#from sklearn.metrics import precision_recall_fscore_support as PRF, roc_auc_score as AUC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import BaggingClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical

# settings
DB_COMMIT_TTL = 300        # control 'commit()' frequency
DB_ENABLE_GZIP = True      # enable gzip can slow down I/O cache, but save some disk space
FVMAKER_WORKER = 16        # you can use cpu_count()
FVMAKER_SLEEP = 10         # heartbeat for Q status report
TDTREE_ENABLE = True       # use td-tree in featvec?
TDTREE_MAX_DEPTH = 0       # limit td-tree depth (raw average depth: 10)
MODE_UPDATE = True         # False if bulk precache, True if incrementally build db (ie. interactive)
MODE_DEBUG = True

WINDOW_TITLE = "Code Viewer"
WINDOW_SIZE = (750, 680)

# consts
BASE_PATH = path.dirname(path.abspath(__file__))
TAR_FILE = path.join(BASE_PATH, 'MG1933029_蒋松儒.xz')
DATA_PATH = path.join(BASE_PATH, 'dataset')
DATA_FILE = path.join(DATA_PATH, 'nju-introdm20.zip')
DATA_PICKLE_FILE = path.join(DATA_PATH, 'data.pkl')
DB_FILE = path.join(DATA_PATH, 'featvec.db')
MODEL_FILE = path.join(DATA_PATH, 'model.pkl')
RESULT_FILE_SCHEMA = path.join(DATA_PATH, 'sample_submission.csv')
RESULT_FILE = path.join(DATA_PATH, 'result.csv')

# trancedental knowledge data
STDLIB_FUNCTIONS = {            # frequent used library function names
  'libio': {
    'cout', 'printf', 'sprintf', 'puts', 'putchar', 'putc', 
    'cin',' scanf', 'sscanf', 'gets', 'getchar', 'getc', 'getline',
  },
  'libmem': {
    'malloc', 'alloc', 'realloc', 'free',
  },
  'libstr': {
    'strlen', 'strcmp', 'strcpy', 'strcat', 'strsub', 'strchr', 'strstr',
    'memcpy', 'memset',
  },
  'libmath': {
    'sqrt', 'pow', 'powf', 'exp', 'log', 'log10', 
    'fabs', 'floor', 'ceil', 'sin', 'cos', 'tan',
  },
}

# sys fix & perf count
def open(fp, rw='r', *args, **kwargs):
  from builtins import open as _open
  return ('b' in rw) and _open(fp, rw, *args, **kwargs) or _open(fp, rw, encoding='utf8', *args, **kwargs)

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    ret = fn(*args, **kwargs)
    end = time()
    print('[Timer]: %s() costs %.4fs' % (fn.__name__, end - start))
    return ret
  return wrapper

# code digester/analyzer
def recursive(fn):              # decorator for RVVisitor
  def wrapper(vis, node):
    fn(vis, node)
    for c in node: vis.visit(c)
  return wrapper

class RVVisitor(NodeVisitor):   # gen representation vector

  def abstract_reprvec(self, ast):
    self.rv = pd.Series({
      # data
      'consts': defaultdict(set),  # dict of const literals {'type': {'value'}}
      'dims': set(),               # const literals for array dim
      # names
      'ids': set(),     # set of ids, including varname/typename/funcname
      # types
      'char': 0,        # count of type textual: 'char'
      'string': 0,      # count of type textual: 'char[]'/'string'
      'int': 0,         # count of type integral: 'int'/'long'/'short'/'signed'/'unsigned'
      'float': 0,       # count of type numerical: 'float'/'double'
      'cptr': 0,        # count of type pointer: 'char*'
      'dptr': 0,        # count of type pointer: 'int*'/'long*'/'short*'/'signed*'/'unsigned*'
      'fptr': 0,        # count of type pointer: 'float*/double*'
      'xptr': 0,        # count of type pointer: 'void*/<struct>*'
      'arr1d': 0,       # count of type 1d-array[]
      'arr2d': 0,       # count of type 2d-matrx[][]
      'arrnd': 0,       # count of type 3d-cube[][][] and dims above
      'object': 0,      # count of type composite: 'struct'/'union' 
      # ops: assignment
      '=': 0,           # count of assign: '='/'<op>='
      # ops: address & sizeof
      'sz': 0,          # count of op: sizeof
      '&v': 0,          # count of op: '&var'
      '*p': 0,          # count of op: '*ptr'
      # ops: arithmetic
      '++': 0,          # count of op: '++'/'p++'
      '--': 0,          # count of op: '--'/'p--'
      '-0': 0,          # count of op: unary '-'
      '+': 0,           # count of op: '+'/'+='
      '-': 0,           # count of op: '-'/'-='
      '*': 0,           # count of op: '*'/'*='
      '/': 0,           # count of op: '/'/'/='
      '%': 0,           # count of op: '%'/'%='
      'arithm': 0,
      # ops: bitwise
      '~': 0,           # count of op: '~'
      '&': 0,           # count of op: '&'/'&='
      '|': 0,           # count of op: '|'/'|='
      '^': 0,           # count of op: '^'/'^='
      '>>': 0,          # count of op: '>>'/'>>='
      '<<': 0,          # count of op: '<<'/'<<='
      'bit': 0,
      # ops: logic
      '!': 0,           # count of op: '!'
      '||': 0,          # count of op: '||'
      '&&': 0,          # count of op: '&&'
      'logic': 0,
      # ops: compare      
      '=!=': 0,         # count of op eqv: '=='/'!='
      '<=>': 0,         # count of op cmp: '>'/'<'/'>='/'<='
      'comp': 0,
      # control flows
      'cond': 0,        # count of branches: 'if'/'case'/'default'/'?:'
      'loop': 0,        # count of loop blocks: 'for'/'while'/'goto'
      # funcall
      'cast': 0,        # count of type cast funcall
      'funcall': 0,     # count of any funcall
      # funcall： lib
      'libio': 0,       # count of funcall: libio
      'libmem': 0,      # count of funcall: libmem
      'libstr': 0,      # count of funcall: libstr
      'libmath': 0,     # count of funcall: libmath
    })    
    super().visit(ast)

    # sum of certain group
    self.rv['arithm'] = sum(self.rv[it] for it in ['++', '--', '-0', '+', '-', '*', '/', '%'])
    self.rv['bit'] = sum(self.rv[it] for it in ['~', '&', '|', '^', '>>', '<<'])
    self.rv['logic'] = sum(self.rv[it] for it in ['!', '||', '&&'])
    return self.rv

  # decl: for id & type
  def visit_ID(self, node): self.rv['ids'].add(node.name)
  def visit_TypeDecl(self, node): self.rv['ids'].add(node.declname)
  def visit_Struct(self, node): self.rv['object'] += 1
  def visit_Union(self, node): self.rv['object'] += 1

  def visit_IdentifierType(self, node):
    types = set(node.names)
    if 'double' in types or 'float' in types: self.rv['float'] += 1
    elif 'char' in types: self.rv['char'] += 1
    elif types & {'int', 'long', 'short', 'unsigned', 'signed'} : self.rv['int'] += 1

  def visit_PtrDecl(self, node):
    try:
      while node and type(node) != IdentifierType:    # deep into the basic type
        node = node.type
      types = set(node.names)
      if 'double' in types or 'float' in types: self.rv['fptr'] += 1
      elif 'char' in types: self.rv['cptr'] += 1
      elif types & {'int', 'long', 'short', 'unsigned', 'signed'} : self.rv['dptr'] += 1
      else: self.rv['xptr'] += 1
    except:
      self.rv['xptr'] += 1

  def visit_ArrayDecl(self, node):
    def _parse_array_decl(node, dim):
      if hasattr(node, 'dim') and type(node.dim) == Constant:
        self.rv['dims'].add(node.dim.value)

      nodtyp = node.type
      if type(nodtyp) == TypeDecl:       # 1d-array
        self.rv['ids'].add(nodtyp.declname)
        if type(nodtyp.type) == IdentifierType and ('char' in nodtyp.type.names) and dim == 1:
          self.rv['string'] += 1
        else:
          if dim == 1: self.rv['arr1d'] += 1 
          elif dim == 2: self.rv['arr2d'] += 1 
          else: self.rv['arrnd'] += 1 
      elif type(nodtyp) == ArrayDecl:    # nd-array
        _parse_array_decl(nodtyp, dim + 1)
      elif type(nodtyp) == PtrDecl:      # forward pointer
        _parse_array_decl(nodtyp, dim)
    
    _parse_array_decl(node, 1)

  # op: for operator
  @recursive
  def visit_Assignment(self, node):
    self.rv['='] += 1
    op = node.op
    if len(op) > 1: self.rv[op[:-1]] += 1   # '<op>='
  @recursive
  def visit_UnaryOp(self, node):
    op = node.op
    if op in ['++', 'p++']: self.rv['++'] += 1
    elif op in ['--', 'p--']: self.rv['--'] += 1
    elif op == '-': self.rv['-0'] += 1
    elif op == '&': self.rv['&v'] += 1
    elif op == '*': self.rv['*p'] += 1
    elif op == 'sizeof': self.rv['sz'] += 1
    elif op in ['!', '~']: self.rv[op] += 1
  @recursive
  def visit_BinaryOp(self, node):
    op = node.op
    if op in ['+', '-', '*', '/', '%', 
              '|', '&', '^', '>>', '<<', 
              '||', '&&']: self.rv[op] += 1
    elif op in ['==', '!=']: self.rv['=!='] += 1
    elif op in ['>', '>=', '<', '<=']: self.rv['<=>'] += 1
  
  # const: for literal value
  def visit_Constant(self, node): self.rv['consts'][node.type].add(node.value)
 
  # control flow: for keywords
  @recursive
  def visit_If(self, node):
    if type(node.iftrue) != If: self.rv['cond'] += 1
    if type(node.iffalse) != If: self.rv['cond'] += 1
  @recursive
  def visit_Case(self, node): self.rv['cond'] += 1
  @recursive
  def visit_Default(self, node): self.rv['cond'] += 1
  @recursive
  def visit_TernaryOp(self, node): self.rv['cond'] += 2
  @recursive
  def visit_For(self, node): self.rv['loop'] += 1
  @recursive
  def visit_While(self, node): self.rv['loop'] += 1
  @recursive
  def visit_DoWhile(self, node): self.rv['loop'] += 1
  @recursive
  def visit_Goto(self, node): self.rv['loop'] += 1
  
  # funcall
  @recursive
  def visit_Cast(self, node): self.rv['cast'] += 1
  @recursive
  def visit_FuncCall(self, node):
    def _find_fn(node):
      if type(node) == ID:
        return node.name
      else:
        for c in node:
          return _find_fn(c)

    self.rv['funcall'] += 1
    fn = _find_fn(node)
    for lib, fns in STDLIB_FUNCTIONS.items():
      if fn in fns:
        self.rv[lib] += 1
        break

KEY_NODE_TYPES = [
  FileAST,
  Decl, FuncDef,
  Compound,
  If, Switch, Case, Default,
  For, While, DoWhile, 
]

class TDVisitor:                # gen AST-tree descriptor

  def abstract_treedescptr(self, ast) -> pd.Series:
    self.height = 0
    self.td = self.preorder(ast, 1)
    return pd.Series({'td': self.td, 'td_ht': self.height})

  def preorder(self, node, depth):
    if depth > self.height: self.height = depth
    root = Node(type(node).__name__)
    if type(node) in KEY_NODE_TYPES and (TDTREE_MAX_DEPTH == 0 or depth < TDTREE_MAX_DEPTH):
      for c in node:
        root.addkid(self.preorder(c, depth+1))
        #self.preorder(c, depth+1)
    return root

class CodeDigester:             # abstract reprvec from src code

  def __init__(self):
    self.parser = CParser()
    self.visitor_rv = RVVisitor()
    self.visitor_td = TDVisitor()
  
  def abstract_reprvec(self, cid:str) -> pd.Series:
    try:
      src = nlist.get(cid)
      ast = self.parser.parse(src)
      id = pd.Series({'id': cid})
      rv = self.visitor_rv.abstract_reprvec(ast)
      td = self.visitor_td.abstract_treedescptr(ast)
      return pd.concat([id, rv, td], copy=False)
    except Exception as e:
      print('[digest] error parsing %r: %r' % (cid, e))
      return None

# code viewer (GUI)
def require_item_selected(fn):  # decorator for CodeViewer
  def wrapper(cv, *args, **kwargs):
    items = cv.tv.selection()
    if not items: return
    cid = cv.tv.item(items[0], 'values')[0]
    
    # lazy load, see 'CodeViewer.setup_workspace.task'
    if cid.startswith('[Test'):
      if None in cv.group_loaded: return
      cv.group_loaded.add(None)           # mark loaded

      grphd = cv.tv_sect_id[None]         # find that sect_id in self.tv
      grp = cv.data['test']
      for i, cid in enumerate(grp.keys()):
        vals = (cid,)
        cv.tv.insert(grphd, i, values=vals)
    elif cid.startswith('[Group'):
      gid = cv.GROUP_HEAD_REGEX.findall(cid)[0]
      if gid in cv.group_loaded: return
      cv.group_loaded.add(gid)            # mark loaded

      grphd = cv.tv_sect_id[gid]          # find that sect_id in self.tv
      grp = cv.data['train'][gid]         # aka. src group { 'cid': 'src' }
      for i, cid in enumerate(grp.keys()):
        vals = (cid,)
        cv.tv.insert(grphd, i, values=vals)
    # show code when a src item selected
    else: fn(cv, cid, *args, **kwargs)
  return wrapper

class CodeViewer:               # GUI code viewer for intuition

  def __init__(self):
    self.clf = joblib.load(MODEL_FILE)   # clf
    self.fvmanager = FVManager()         # FVManager
    self.digester = CodeDigester()       # CodeDigester

    self.data = self.fvmanager.data
    self.group_loaded = set()     # { 'gid' }, for lazy load marking loaded, 'gid==None' for ds_ts
    self.tv_sect_id = { }         # { 'gid': sect_id }, when lazy load for finding section to insert
    self.GROUP_HEAD_REGEX = REGEX(r'\[Group-(\w*)\]')  # for lazy load extracting 'gid'
    
    self.setup_gui()
    self.setup_workspace()
    
    try: tk.mainloop()
    except KeyboardInterrupt: pass
  
  def setup_gui(self):
    # root window
    wnd = tk.Tk()
    wnd.title(WINDOW_TITLE)
    (wndw, wndh), scrw, scrh = WINDOW_SIZE, wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    wnd.geometry('%dx%d+%d+%d' % (wndw, wndh, (scrw - wndw) // 2, (scrh - wndh) // 4))
    wnd.resizable(False, False)

    # main menu bar
    menu = tk.Menu(wnd, tearoff=False)
    menu.add_command(label="View at left side", command=lambda: self._clt_cv('up'))
    menu.add_command(label="View at right side", command=lambda: self._clt_cv('down'))

    # top: main panel
    frm11 = ttk.Frame(wnd)
    frm11.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
    if True:
      # left: group tree
      frm21 = ttk.Frame(frm11)
      frm21.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
      if True:
        sb = ttk.Scrollbar(frm21)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        cols = { 'cid':  ("cid", 150, tk.W) }
        tv = ttk.Treeview(frm21, show=['headings'],    # table-like
                          columns=list(cols.keys()),
                          selectmode=tk.BROWSE, yscrollcommand=sb.set)
        sb.config(command=tv.yview)
        for k, v in cols.items():
          tv.column(k, width=v[1], anchor=v[2])
          tv.heading(k, text=v[0])
        clt_show_cvU = lambda evt: self._clt_cv('up')
        tv.bind("<Double-Button-1>", clt_show_cvU)
        tv.bind('<Return>', clt_show_cvU)
        tv.bind('<Left>', clt_show_cvU)
        tv.bind('1', clt_show_cvU)
        clt_show_cvD = lambda evt: self._clt_cv('down')
        tv.bind("<Button-3>", clt_show_cvD)
        tv.bind('<Right>', clt_show_cvD)
        tv.bind('2', clt_show_cvD)
        tv.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
        self.tv = tv
      
      # right: code viewer
      frm22 = ttk.Frame(frm11)
      frm22.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
      if True:
        frm31 = ttk.Frame(frm22)
        frm31.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        if True:
          ttk.Label(frm31, text='Cid1: ').pack(side=tk.LEFT, padx=4)

          var = tk.StringVar(wnd)
          self.var_xid = var
          ent = ttk.Entry(frm31, textvariable=var)
          ent.bind('<Return>', lambda evt: self._clt_tx('up'))
          ent.pack(side=tk.LEFT, padx=4)

          ttk.Label(frm31, text='Cid2: ').pack(side=tk.LEFT, padx=4)

          var = tk.StringVar(wnd)
          self.var_yid = var
          ent = ttk.Entry(frm31, textvariable=var)
          ent.bind('<Return>', lambda evt: self._clt_tx('down'))
          ent.pack(side=tk.LEFT, padx=4)

          ttk.Button(frm31, text='Judge!', command=self.judge).pack(side=tk.LEFT, padx=4)

        frm32 = ttk.Frame(frm22)
        frm32.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
        if True:
          txt = ScrolledText(frm22)
          txt.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
          self.cvU = txt
          txt = ScrolledText(frm22)
          txt.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
          self.cvD = txt

    # bottom: status bar
    frm12 = ttk.Frame(wnd)
    frm12.pack(side=tk.BOTTOM, fill=tk.X, expand=tk.YES)
    if True:
      var = tk.StringVar(wnd, "[OK] Init.")
      self.var_stat_msg = var
      ttk.Label(frm12, textvariable=var).pack(fill=tk.X, expand=tk.YES)

  def setup_workspace(self):
    # test set: lazy load, see 'require_item_selected'
    vals = ("[Test]")
    grphd = self.tv.insert('', tk.END, values=vals, open=False)
    self.tv_sect_id[None] = grphd    # 'gid==None' for ds_ts
    
    # train set: lazy load, see 'require_item_selected'
    for gid in self.data['train'].keys():
      vals = (f"[Group-{gid}]")
      grphd = self.tv.insert('', tk.END, values=vals, open=False)
      self.tv_sect_id[gid] = grphd

  def _clt_show_code(self, cid, where):
    src = nlist[cid]
    if where == 'up':
      self.var_xid.set(cid)
      self.cvU.delete(1.0, tk.END)
      self.cvU.insert(tk.END, src)
    elif where == 'down':
      self.var_yid.set(cid)
      self.cvD.delete(1.0, tk.END)
      self.cvD.insert(tk.END, src)
    else: raise ValueError
    if MODE_DEBUG: print(self.fvmanager.get_or_make_reprvec(cid))

  @require_item_selected
  def _clt_cv(self, cid, where='up'):
    self._clt_show_code(cid, where)

  def _clt_tx(self, where='up'):
    if where == 'up': cid = self.var_xid.get()
    elif where == 'down': cid = self.var_yid.get()
    else: raise ValueError
    if cid in nlist:
      self._clt_show_code(cid, where)
    else:
      self.var_stat_msg.set('[Error] Invalid cid: %r' % cid)

  def judge(self):
    xid, yid = self.var_xid.get(), self.var_yid.get()
    tid = f'{xid}_{yid}'
    fv = self.fvmanager.get_or_make_featvec(tid, type=FVType.FEATVEC_UNKNOWN_EX)
    fv = fv.drop('id')
    Y = self.clf.predict(pd.DataFrame(fv).T)
    r = np.argmax(Y)   # keras
    
    self.var_stat_msg.set('[Judge] result is %s for %r' % (r, tid))

# database & model
db = None
db_lock = RLock()
db_count = DB_COMMIT_TTL       # action count for bulk insert
Model = declarative_base()

def setup(dbfile=DB_FILE):
  global db
  engine = sql.create_engine('sqlite:///%s?check_same_thread=False' % dbfile)
  Model.metadata.create_all(engine)
  session_maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
  db = session_maker()

def save(model=None, auto_commit=False):
  global db_count
  with db_lock:
    db_count -= 1
    if model: db.add(model)
    if db_count <= 0 or auto_commit:
      db_count = DB_COMMIT_TTL   # reset TTL
      try: db.commit()
      except Exception as e: print(e)

class FVType(Enum):
  FEATVEC_POSITIVE = 1         # train pos
  FEATVEC_NEGATIVE = 0         # train neg
  FEATVEC_UNKNOWN = -1         # test
  FEATVEC_UNKNOWN_EX = -100    # test extra
  REPRVEC = 100

class FVCache(Model):

  @declared_attr
  def __tablename__(cls): return cls.__name__
  
  sn = sql.Column(sql.INTEGER, primary_key=True, autoincrement=True, comment='dummy pk')
  id = sql.Column(sql.TEXT, comment='use cid/tid in raw data')
  fv = sql.Column(sql.BLOB, comment='(compressed) pickle')
  type = sql.Column(sql.Enum(FVType))

  def __repr__(self):
    return '<%s id=%r type=%r>' % (self.__class__.__name__, self.id, self, type)

def type2trgt(type:FVType) -> int:
  return {
    FVType.FEATVEC_UNKNOWN: None,
    FVType.FEATVEC_UNKNOWN_EX: None,
    FVType.FEATVEC_POSITIVE: 1,
    FVType.FEATVEC_NEGATIVE: 0,
  }[type]

def abstract_featvec(xfv:pd.Series, yfv:pd.Series, trgt:int=None) -> pd.Series:  
  # clean start
  fvd = {'id': f'{xfv["id"]}_{yfv["id"]}'}

  # deal with scalars
  cols = xfv.keys()[4:-2]     # order according to `RVVisitor.abstract_reprvec()` and `TDVisitor.abstract_treedescptr()`
  for col in cols:
    fvd[col] = abs(xfv[col] - yfv[col])
    
  # deal with sets: ids
  xids = {id.upper() for id in (xfv['ids'] - STDLIB_FUNCTIONS['libio']) if id}
  yids = {id.upper() for id in (yfv['ids'] - STDLIB_FUNCTIONS['libio']) if id}
  ncomm  = len(xids & xids)
  ntotal = len(yids | yids)
  fvd['ids'] = ncomm
  fvd['ids_rt'] = ntotal and (100 * ncomm // ntotal) or 0
  
  # deal with sets: consts
  def collect_const(cst, fv):
    for k, st in fv['consts'].items():
      if k in ['int', 'long', 'short', 'signed', 'unsigned']:
        for v in st:
          try: cst.add(int(v))
          except ValueError: cst.add(v)
      elif k == 'char':
        def to_int(chr):
          if len(chr) == 1:
            return ord(chr)
          if len(chr) == 2:
            UNESCAPE_DICT = {
              '\\n': '\n',
              '\\r': '\r',
              '\\t': '\t',
              '\\0': '\0',
              '\\f': '\f',
              "\\'": "\'",
              '\\"': '\"',
            }
            return (chr in UNESCAPE_DICT) and ord(UNESCAPE_DICT[chr]) or chr
          else: return chr
        for v in st: cst.add(to_int(v[1:-1]))    # remove '', convert to int
      elif k == 'string':
        for v in st: 
          s = v[1:-1]         # remove ""
          if s[:1] in [',', '-', ':']:
            cst.add(s[:1])
            s = s[1:]
          elif s[:2] in ['\\n', '\\t']:
            cst.add(s[:2])
            s = s[2:]
          if s[-1:] in [',', '-', ':']:
            cst.add(s[-1:])
            s = s[:-1]
          elif s[-2:] in ['\\n', '\\t']:
            cst.add(s[-2:])
            s = s[:-2]
      else:                 # float or other
        for v in st: cst.add(v)
    if 'endl' in fv['ids']: # fix 'endl' == '\n'
      fv['ids'].remove('endl')
      cst.add(ord('\n'))

  xconsts, yconsts = set(), set()
  collect_const(xconsts, xfv)
  collect_const(yconsts, yfv)
  ncomm  = len(xconsts & yconsts)
  ntotal = len(xconsts | yconsts)
  fvd['consts'] = ncomm
  fvd['consts_rt'] = ntotal and (100 * ncomm // ntotal) or 0
  
  # deal with sets: dims
  xdims = {int(dim) // 100 for dim in xfv['dims'] if dim.isdigit()}
  ydims = {int(dim) // 100 for dim in yfv['dims'] if dim.isdigit()}
  ncomm  = len(xdims & ydims)
  ntotal = len(xdims | ydims)
  fvd['dims'] = ncomm
  fvd['dims_rt'] = ntotal and (100 * ncomm // ntotal) or 0
  
  # deal with ast
  fvd['td'] = TDTREE_ENABLE and simple_distance(xfv['td'], yfv['td']) or 0
  try:
    fvd['td_hd'] = TDTREE_ENABLE and abs(xfv['td_ht'] - yfv['td_ht']) or 0
  except:
    set_trace()

  # deal with label
  if trgt is not None: fvd['class'] = trgt     # whether train or test?
  
  # form a pd.Series
  return pd.Series(fvd)

class FVMaker(Thread):

  def __init__(self, tid, fvm):
    super().__init__(name='DownloadWorker-%d' % tid, daemon=True)
    self.tid = tid
    self.fvm = fvm
    self.Q = fvm.Q
    self.evt_stop = fvm.evt_stop
    self.digester = CodeDigester()    # one thread per digester

  def run(self):
    while not self.evt_stop.is_set():
      id, type = None, None
      while not self.evt_stop.is_set() and id is None:
        try: id, type = self.Q.get(timeout=FVMAKER_SLEEP)
        except Empty: pass
      if not id: continue

      if type == FVType.REPRVEC:
        rv = self.digester.abstract_reprvec(id)
        self.fvm.put(id, rv, type=type)         # 'rv' might be None
      else:
        try:
          xid, yid = id.split('_')
          xfv, yfv = self.fvm.get(xid), self.fvm.get(yid)
          if xfv is not None and yfv is not None:
            fv = abstract_featvec(xfv, yfv, type2trgt(type))
            self.fvm.put(id, fv, type=type)
          else:
            print('[%s] reprvec missing for tid %r' % (self.__class__.__name__, id))
        except Exception as e:
          print('[%s-%s] failed for tid %r: %r' % (self.__class__.__name__, self.tid, id, e))

      # if MODE_DEBUG: print('[%s-%s] done for %s' % (self.__class__.__name__, self.tid, id))
      self.Q.task_done()

def wait_for_complete(fn):      # decorator for FVManager
  @timer
  def wrapper(fvm, *args, **kwargs):
    fn(fvm, *args, **kwargs)
    
    while fvm.Q.qsize():
      last_qsize = fvm.Q.qsize()
      sleep(FVMAKER_SLEEP)
      cur_qsize = fvm.Q.qsize()
      speed = (last_qsize - cur_qsize) / FVMAKER_SLEEP
      print('[Queue] pending %s tasks (speed: %.2f, remaining %ss)' % (cur_qsize, speed, speed and (cur_qsize // speed) or 'N/A'))
    sleep(FVMAKER_SLEEP)
    fvm.evt_stop.set()
  return wrapper

class FVManager:

  FV_LOCAL_CACHE = { }
  
  def __init__(self, worker=FVMAKER_WORKER):
    setup()
    self.data = get_data()
    self.digester = CodeDigester()
    self.evt_stop = Event()
    self.Q = Queue()
    self.workers = [FVMaker(i+1, self) for i in range(worker)]
    for worker in self.workers: worker.start()
  
  def __del__(self):
    self.evt_stop.set()
    for worker in self.workers: worker.join()
    save(auto_commit=True)
    db.close()

  # low level API
  def _pack_fv(self, fv:pd.Series) -> bytes:
    fvpkl = pickle.dumps(fv, protocol=4)
    if DB_ENABLE_GZIP:
      bio = BytesIO()
      with gzip.GzipFile(mode='wb', fileobj=bio) as fh:
        fh.write(fvpkl)
      return bio.getvalue()
    else:
      return fvpkl
  def _unpack_fv(self, fv:bytes) -> pd.Series:
    if DB_ENABLE_GZIP:
      bio = BytesIO(fv)
      with gzip.GzipFile(mode='rb', fileobj=bio) as fh:
        fvpkl = fh.read()
      return pickle.loads(fvpkl)
    else:
      return pickle.loads(fv)

  def get(self, id:str) -> pd.Series:     # return None if not cached
    # load from local cache
    if id in self.FV_LOCAL_CACHE:
      return self.FV_LOCAL_CACHE[id]
    
    # load from db
    fvc = db.query(FVCache.fv).filter_by(id=id).one_or_none()
    if not fvc: return None
    fv = self._unpack_fv(fvc[0])
    
    # save local cache
    self.FV_LOCAL_CACHE[id] = fv
    return fv
  
  def put(self, id:str, fv:pd.Series, type:FVType):
    bfv = self._pack_fv(fv)
    if MODE_UPDATE:
      fvc = db.query(FVCache).filter_by(id=id).one_or_none()
      if fvc: fvc.fv = bfv
      else: fvc = FVCache(id=id, fv=bfv, type=type)
    else:
      fvc = FVCache(id=id, fv=bfv, type=type)
    
    # save to db
    save(fvc)
    # update local cache
    self.FV_LOCAL_CACHE[id] = fv

  def exists(self, id:str) -> bool:
    return bool(db.query(FVCache).filter_by(id=id).count())

  def count_train(self) -> int:
    return db.query(FVCache).filter(or_(FVCache.type == FVType.FEATVEC_POSITIVE, 
                                        FVCache.type == FVType.FEATVEC_NEGATIVE)).count()

  # high level API (read)
  def get_or_make_reprvec(self, cid:str) -> pd.Series:    # auto make if not cached
    rv = self.get(cid)
    if rv is None:
      rv = self.digester.abstract_reprvec(cid)
      self.put(cid, rv, type=type)
    return rv

  def get_or_make_featvec(self, tid:str, type:FVType=FVType.FEATVEC_UNKNOWN) -> pd.Series:  # auto make if not cached
    fv = self.get(tid)
    if fv is None:
      xid, yid = tid.split('_')
      xfv, yfv = self.get(xid), self.get(yid)
      fv = abstract_featvec(xfv, yfv, trgt=type2trgt(type))
      self.put(tid, fv, type=type)
    return fv

  # higher level API (bulk write)
  @wait_for_complete
  def build_reprvec(self):
    with db_lock: rvids = {res[0] for res in db.query(FVCache.id).filter_by(type=FVType.REPRVEC).all()}
    for cid in nlist.keys():
      if cid not in rvids:      # if not cached
        self.Q.put((cid, FVType.REPRVEC))
  
  @wait_for_complete
  def build_featvec_test(self):
    self.precache_reprvec()     # load rvs
    with db_lock: fvids = {res[0] for res in db.query(FVCache.id).filter_by(type=FVType.FEATVEC_UNKNOWN).all()}
    with open(RESULT_FILE_SCHEMA) as fin:
      fin.readline()                  # ignore headers
      for line in fin.read().split('\n'):
        tid = line.split(',')[0]
        if tid not in fvids:    # if not cached
          self.Q.put((tid, FVType.FEATVEC_UNKNOWN))
  
  @wait_for_complete
  def build_featvec_train(self, nlimit=10000):
    if type(nlimit) == tuple: NPOS, NNEG = nlimit[0], nlimit[1]
    else: NPOS, NNEG = nlimit, nlimit

    self.precache_reprvec()     # load rvs
    ds_tr = self.data['train']
    gids, gid_cids = list(ds_tr.keys()), { }   # cache key lists

    # sample from same group, class = 1
    with db_lock: fvids = {res[0] for res in db.query(FVCache.id).filter_by(type=FVType.FEATVEC_POSITIVE).all()}
    for _ in range(NPOS):
      gid = choice(gids)
      if gid not in gid_cids:
        gid_cids[gid] = list(ds_tr[gid].keys())

      xid, yid = sample(gid_cids[gid], 2)
      tid = f'{xid}_{yid}'
      if tid not in fvids:      # if not cached
        self.Q.put((tid, FVType.FEATVEC_POSITIVE))

    # sample from diff group, class = 0
    with db_lock: fvids = {res[0] for res in db.query(FVCache.id).filter_by(type=FVType.FEATVEC_NEGATIVE).all()}
    for _ in range(NNEG):
      gid1, gid2 = sample(gids, 2)
      if gid1 not in gid_cids: gid_cids[gid1] = list(ds_tr[gid1].keys())
      if gid2 not in gid_cids: gid_cids[gid2] = list(ds_tr[gid2].keys())

      xid, yid = choice(gid_cids[gid1]), choice(gid_cids[gid2])
      tid = f'{xid}_{yid}'
      if tid not in fvids:      # if not cached
        self.Q.put((tid, FVType.FEATVEC_NEGATIVE))

  # higher level API (bulk read)
  @timer
  def precache_reprvec(self):
    rvs = db.query(FVCache).filter_by(type=FVType.REPRVEC).all()
    for rv in rvs: self.FV_LOCAL_CACHE[rv.id] = self._unpack_fv(rv.fv)
  
  @timer
  def precache_test_featvec(self):
    rvs = db.query(FVCache).filter_by(type=FVType.FEATVEC_UNKNOWN).all()
    for rv in rvs: self.FV_LOCAL_CACHE[rv.id] = self._unpack_fv(rv.fv)
  
  @timer
  def load_reprvec(self) -> pd.DataFrame:
    fvs = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.REPRVEC).all()]
    return pd.concat(fvs, axis=1, copy=False).T
  
  @timer
  def load_train_featvec(self, nlimit=0) -> pd.DataFrame:   # randomly return bulk of train featvec
    if not nlimit:
      fvs_pos = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_POSITIVE).all()]
      fvs_neg = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_NEGATIVE).all()]
    else:
      fvs_pos = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_POSITIVE).limit(nlimit//2).all()]
      fvs_neg = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_NEGATIVE).limit(nlimit//2).all()]

    try:
      df_pos = pd.concat(fvs_pos, axis=1, copy=False).T
      df_neg = pd.concat(fvs_neg, axis=1, copy=False).T
      #set_trace()
      return pd.concat([df_pos, df_neg], copy=False)
    except: return None

  @timer
  def load_test_featvec(self, nlimit=0) -> pd.DataFrame:
    if not nlimit:
      fvs = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_UNKNOWN).all()]
    else:
      fvs = [self._unpack_fv(fvc.fv) for fvc in db.query(FVCache).filter_by(type=FVType.FEATVEC_UNKNOWN).limit(nlimit).all()]

    return pd.concat(fvs, axis=1, copy=False).T

# data preprocess & train & test
nlist = None    # global src index dict, aka. data['nlist']

def normalize(df):
  df = df.copy()
  for col in df.columns[:-1]:     # ignore taget label
    avg, std = df[col].mean(), df[col].std()
    df[col] = df[col].apply(lambda x: std and ((x - avg) / std) or 0)
  return df

def ten_fold(df, shuffle=True): # use as generator
  if shuffle: df = df.sample(frac=1)
  SILCE = int(np.ceil(len(df) / 10))
  for i in range(10):
    cp1, cp2 = i*SILCE, (i+1)*SILCE
    df_ts = df[cp1:cp2]
    df_tr = pd.concat([df[:cp1], df[cp2:]], copy=False)
    yield (df_tr, df_ts)

@timer
def get_data() -> dict:         # raw data to dict format
  '''
    data = { 'train': ds_tr, 'test': ds_ts, 'nlist': nlist }
      ds_tr = { 'gid': { 'cid': src } }
      ds_ts = { 'cid': src }
      nlist = { 'cid': src }
  '''

  global nlist

  if path.exists(DATA_PICKLE_FILE):
    data = joblib.load(DATA_PICKLE_FILE)
  else:
    GID_CID_REGEX = REGEX('train/train/(\w*)/(\w*).txt')
    CID_REGEX = REGEX('test/test/(\w*).txt')
    ds_tr, ds_ts, nlist = defaultdict(dict), { }, { }

    zip = ZipFile(DATA_FILE)
    for fp in zip.namelist():
      if not fp.endswith('.txt'): continue
      with zip.open(fp) as fh:
        src = fh.read().decode()
    
      if fp.startswith('test/test/'):
        cid = CID_REGEX.findall(fp)[0]
        ds_ts[cid] = src
      elif fp.startswith('train/train/'):
        gid, cid = GID_CID_REGEX.findall(fp)[0]
        ds_tr[gid][cid] = src
      if cid in nlist: print('[get_data] duplicated cid: %r' % cid)
      nlist[cid] = src
    
    data = { 'train': ds_tr, 'test': ds_ts, 'nlist': nlist }
    joblib.dump(data, DATA_PICKLE_FILE, compress=('xz', 9), protocol=4)
  
  nlist = data['nlist']          # assign to global
  return data

def P_R_F(TP, FN, FP, TN):
  P = TP / (TP + FP)
  R = TP / (TP + FN)
  F = 2 * TP / (2 * TP + FP + FN)
  return P, R, F

@timer
def run(df, clf):
  P, R, F, A = 0, 0, 0, 0

  for df_tr, df_ts in ten_fold(df):
    # train
    X, y = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    clf = clf.fit(X, y)
    # test
    X, y = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]
    y_pred = clf.predict(X)
    # analysis
    p, r, f, _ = PRF(y, y_pred, average='macro', zero_division=0)
    a = AUC(y, y_pred)
    P, R, F, A = P + p, R + r, F + f, A + a
  
  P, R, F, A = P * 10, R * 10, F * 10, A * 10
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%, AUC: %.2f%%' % (P, R, F, A))
  return P, R, F, A

@timer
def run_keras(df) -> object:
  # model
  NFEAT, NCLASS = len(df.columns) - 1, 2
  model = Sequential([
    Dense(64, activation='relu', input_dim=NFEAT),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NCLASS, activation='softmax'),
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  
  # train
  TP, FN, FP, TN = 0, 0, 0, 0    # confusion matrix
  for i, (df_tr, df_ts) in enumerate(ten_fold(df)):
    Xtrain, ytrain = df_tr[df_tr.columns[:-1]], df_tr[df_tr.columns[-1]]
    Ytrain = to_categorical(ytrain, NCLASS)       # vectorize: 0 => [1,0], 1 => [0,1]
    Xtest, ytest = df_ts[df_ts.columns[:-1]], df_ts[df_ts.columns[-1]]
    print('[train] round %s' % i)
    model.fit(Xtrain, Ytrain, batch_size=32, epochs=10, verbose=True)
    Ypred = model.predict(Xtest)
    for i, Y in enumerate(Ypred):
      y = np.argmax(Y)           # take the most probable class's id
      if y == ytest.iloc[i]:
        if y == 1: TP += 1       # 1 = positive, 0 = negative
        else: TN += 1
      else:
        if y == 1: FP += 1
        else: FN += 1
  
  P, R, F = P_R_F(TP, FN, FP, TN)
  P, R, F = P * 100, R * 100, F * 100
  print('pres: %.2f%%, recall: %.2f%%, fscore: %.2f%%' % (P, R, F))
  return model, (P, R, F)

@timer
def train(n):
  #clfs = [
  #  # Decision Tree
  #  DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split=3, min_samples_leaf=2),
  #  # SVM
  #  SVC(C=0.7, cache_size=512),
  #  # kNN
  #  KNeighborsClassifier(n_neighbors=5, weights='distance', p=1, n_jobs=FVMAKER_WORKER),
  #  # Bagging
  #  BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5, weights='distance', p=1, n_jobs=FVMAKER_WORKER),
  #                    n_estimators=7, max_samples=0.5, max_features=0.5),
  #]

  # data
  fvm = FVManager()
  #df_rv = fvm.load_reprvec()
  df = fvm.load_train_featvec(n)
  #set_trace()
  del df['id']
  df_norm = df           # normalize(df)
  print('[train] using %s samples' % len(df))

  perf = [ ]    # [ (F-score, clf) ]
  # train (sklearn)
  #for clf in clfs:
  #  clf_name = type(clf).__name__
  #  print(f'<< [Clf] using {clf_name}.')
  #  try:
  #    _, _, F, _ = run(df_norm, clf)
  #    perf.append((F, clf))
  #  except Exception as e:
  #    print('[Error] ' + str(e))
  
  # train (keras)
  print(f'<< [Clf] using NN.')
  clf, (_, _, F) = run_keras(df_norm)
  perf.append((F, clf))

  # save
  perf.sort(reverse=True)
  best_model = perf[0][1]
  joblib.dump(best_model, MODEL_FILE, compress=('xz', 9), protocol=4)
  print('[train] model saved')

@timer
def test_sklearn():
  clf = joblib.load(MODEL_FILE)
  df = FVManager().load_test_featvec()
  with open(RESULT_FILE_SCHEMA) as fin, open(RESULT_FILE, 'w+') as fout:
    fout.write(fin.readline())      # copy headers
    
    tids = [ ]
    for i, line in enumerate(fin.read().split('\n')):
      tid = line.split(',')[0]
      tids.append(tid)

    for i, r in enumerate(clf.predict(df)):  # bulk predict
      #if df.iloc[i]['td'] < 10: r = 1        # FIXME: damn empirical value
      fout.write(f'{tids[i]},{r}\n')
      if not i % 1000: fout.flush() # bulk flush

@timer
def test_keras():
  clf = joblib.load(MODEL_FILE)
  fvm = FVManager()
  fvm.precache_test_featvec()
  with open(RESULT_FILE_SCHEMA) as fin, open(RESULT_FILE, 'w+') as fout:
    fout.write(fin.readline())      # copy headers

    fvs = [ ]
    for i, line in enumerate(fin.read().split('\n')):
      tid = line.split(',')[0]
      fv = fvm.get(tid)
      if fv is not None:
        fv['id'] = tid       # FIXME: temporarily fix the old sorted() problem
      fvs.append(fv)
    
    df = pd.concat(fvs, axis=1, copy=False).T
    tids = df[df.columns[:1]]
    y = pd.Series([np.argmax(Y) for Y in clf.predict(df[df.columns[1:]])])     # take the most probable class's id
    df = pd.concat([tids, y], axis=1, copy=False)
    for _, row in df.iterrows():
      tid = row['id']
      r = row[0]        # column name for y is int '0'
      fout.write(f'{tid},{r}\n')

@timer
def make_tar():
  with ZipFile(TAR_FILE, mode='w', compression=ZIP_LZMA) as fh:
    FP_NAME = path.splitext(path.basename(TAR_FILE))[0]
    FNS = [
      'README.md',
      'Makefile',
      'clone_detect.py',
      '基于词频统计和抽象语法树对比的克隆代码检测方法.doc',
      '基于词频统计和抽象语法树对比的克隆代码检测方法.pdf',
      'dataset/data.pkl',
      'dataset/featvec.db',
      'dataset/model.pkl',
      'dataset/result.csv',
      'dataset/sample_submission.csv',
    ]
    for fn in FNS:
      fh.write(fn, f'{FP_NAME}/{fn}')
      print(f'[tar] file {fn} added')

@timer
def FIX_add_feature_tdht():
  fvm = FVManager()
  
  td_ht = { }   # calc height for each TD-tree
  p = CParser()
  v = TDVisitor()
  for cid in nlist:
    try:
      ast = p.parse(nlist[cid])
      ht = v.abstract_treedescptr(ast)['td_ht']
    except Exception as e:
      ht = 10    # trancedental average value
    td_ht[cid] = ht
  
  def _unpack_fv(fv:bytes) -> pd.Series:
    if DB_ENABLE_GZIP:
      bio = BytesIO(fv)
      with gzip.GzipFile(mode='rb', fileobj=bio) as fh:
        fvpkl = fh.read()
      return pickle.loads(fvpkl)
    else:
      return pickle.loads(fv)
  
  def _pack_fv(fv:pd.Series) -> bytes:
    fvpkl = pickle.dumps(fv, protocol=4)
    if DB_ENABLE_GZIP:
      bio = BytesIO()
      with gzip.GzipFile(mode='wb', fileobj=bio) as fh:
        fh.write(fvpkl)
      return bio.getvalue()
    else:
      return fvpkl
  
  # add td_ht for each reprvec
  fvcs = db.query(FVCache).filter_by(type=FVType.REPRVEC).all()
  totcnt = len(fvcs)
  for i, fvc in enumerate(fvcs):
    fv = _unpack_fv(fvc.fv)
    if fv is None: continue
    fv['td_ht'] = td_ht[fv['id']]
    
    fvc.fv = _pack_fv(fv)    
    save(fvc)
    if i % 5000 == 0:
      print('[REPRVEC] done %s/%s' % (i, totcnt))
  
  # add td_hd for each featvec
  for fvtype in [FVType.FEATVEC_POSITIVE, FVType.FEATVEC_NEGATIVE, FVType.FEATVEC_UNKNOWN]:
    fvcs = db.query(FVCache).filter_by(type=fvtype).all()
    totcnt = len(fvcs)
    for i, fvc in enumerate(fvcs):
      fv = _unpack_fv(fvc.fv)
      if fv is None: continue
      xid, yid = fv['id'].split('_')
      fv['td_hd'] = abs(td_ht[xid] - td_ht[yid])
      
      fvc.fv = _pack_fv(fv)
      save(fvc)
      if i % 5000 == 0:
        print('[%s] done %s/%s' % (fvtype, i, totcnt))

# help info
def help():
  help_text = '''Usage: clone_detect.py
  rv              build all ReprVec (size=5.15w)
  fv_ts           build all test FeatVec (size=20w)
  fv_tr [size]    build random train FeatVec
  tr [size]       train & save model with random FeatVec (size=0 for all)
  ts              load model, predict on test dataset, write result.csv
  cv              open CodeViewr (GUI) for manual check or intuition
'''
  print(help_text)
  exit(0)

if __name__ == '__main__':
  #FIX_add_feature_tdht() ; save(auto_commit=True) ; db.close(); exit(0)
  
  act = len(argv) >= 2 and argv[1]
  n = (len(argv) >= 3 and argv[2].isdigit()) and int(argv[2]) or 0
  n2 = (len(argv) >= 4 and argv[3].isdigit()) and int(argv[3]) or None

  if act == 'cv':
    print('>> start CodeViewer...')
    CodeViewer()
    save(auto_commit=True)
  elif act == 'rv':
    print('>> build ReprVec...')
    MODE_UPDATE = False
    FVManager().build_reprvec()
    save(auto_commit=True)
  elif act == 'fv_ts':
    print('>> build FeatVec for test...')
    MODE_UPDATE = False
    FVManager().build_featvec_test()
    save(auto_commit=True)
  elif act == 'fv_tr':
    print('build FeatVec for train with n = (%s, %s)  ...' % (n, n2))
    MODE_UPDATE = False
    fvm = FVManager()
    if n2 is None: fvm.build_featvec_train(n)
    else:          fvm.build_featvec_train((n, n2))
    save(auto_commit=True)
    cnt = fvm.count_train()
    print('[fv_tr] now you have %d featvec ready to train' % cnt)
  elif act == 'tr':
    print('>> train & save model...')
    train(n)
  elif act == 'ts':
    print('>> test on dataset using clf...')
    test_keras()
  elif act == 'tar':
    print('>> make %s...' % TAR_FILE)
    make_tar()
  else:
    help()
