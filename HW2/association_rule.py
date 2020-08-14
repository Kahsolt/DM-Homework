#!/usr/bin/env python3
# Author: Armit
# Create Time: 2020/04/19 

# 度量：
#  支持度(A) = (包含A的记录数量) / (总的记录数量)
#  置信度(A->B) = (包含A和B的记录数量) / (包含A的记录数量)
#  提升度(A->B) = 置信度(A->B) / (支持度A)

from sys import getsizeof
from os import path
import time
import re
import rarfile
from collections import defaultdict

BASE_DIR = path.dirname(path.abspath(__name__))
DATA_DIR = path.join(BASE_DIR, 'dataset')

CASE_MERGE_PARAM = True
CASE_FILTER = ['**EOF**', 'ls', 'll', 'la', 'rm', 'cd', 'mv', 'cp', 'vi', 'vim', 'exit', 'elm', 'find', 'emacs']
DEBUG = False

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time.time()
    ret = fn(*args, **kwargs)
    end = time.time()
    print('[Timer]: %s() costs %.4fs' % (fn.__name__, end - start))
    return ret
  return wrapper

@timer
def BruteForce(data, min_support):
  items = list({it for tx in data for it in tx})
  nItems, nTrans = len(items), len(data)
  print(f'>> BruteForce(min_support={min_support}) on {nTrans} tx with {nItems} items')
  
  freqset = defaultdict(list)   # {'k': set}
  itr = 0
  while itr < 2**nItems-1:  
    # enumerate next subset
    itr += 1
    c = set()
    for idx, sel in enumerate(bin(itr)[2:].rjust(nItems, '0')):
      if sel == '1':
        c.add(items[idx])

    # filter by min_support
    supp = sum(c.issubset(tx) for tx in data) / nTrans
    if supp >= min_support:
      freqset[len(c)].append(c)
  
  for k in sorted(freqset.keys()):
    print(f'<< freq-{k} sets: {freqset[k]}')

@timer
def Apriori(data, min_support):
  # generate 1-freq
  k = 1
  items = {it for tx in data for it in tx}
  Ck = [{it} for it in items]

  # (k-1) iters
  nItems, nTrans = len(items), len(data)
  print(f'>> Apriori(min_support={min_support}) on {nTrans} tx with {nItems} items')
  while Ck:
    # filter by min_support
    cntr = defaultdict(int)
    for cid, c in enumerate(Ck):
      for tx in data:
        if c.issubset(tx):
          cntr[cid] += 1
    Cfreq = [ ]
    for cid, freq in cntr.items():
      supp = freq / nTrans
      if supp >= min_support:
        Cfreq.append(Ck[cid])
    
    # echo out
    if not Cfreq: break
    print(f'<< freq-{k} sets: {Cfreq}')
    
    # generate (k+1)-freq
    k += 1
    Ckk = [ ]
    for i in range(len(Cfreq)-1):
      for j in range(i+1,len(Cfreq)):
        cc = Cfreq[i] | Cfreq[j]
        if len(cc) == k and cc not in Ckk:
          Ckk.append(cc)
    
    # update
    Ck = Ckk

@timer
def FP_Growth(data, min_support):
  
  class Node:
  
    def __init__(self, value='Ø', count=0, parent=None):
      self.value = value    # item name
      self.count = count    # partial count in a string cluster
      self.next = None      # as linklist
      self.parent = parent  # as tree
      self.children = { }   # as tree, {'item': Node}
    
    def display(self, depth=0):
      print('  ' * depth, self.value, ':', self.count)
      for child in self.children.values():
        child.display(depth + 1)
  
  class Head:
  
    def __init__(self, value, count=0, next=None):
      self.value = value    # item name
      self.count = count    # total count
      self.next = next      # Node
    
    def display(self):
      print(f'{self.value} : {self.count}')
    
  class FPTree:
    
    def __init__(self, root=None, head=None):
      self.root = root    # Node
      self.head = head    # {'item': Head}
    
    def display(self):
      print('[Tree]')
      self.root.display()
      print('[Head]')
      for h in self.head.values(): h.display()
    
    @classmethod
    def create(cls, data, min_count=1):
      # collect 1-frequent, filter by min_count
      cntr = defaultdict(int)   # {'item': count}
      for tx, cnt in data.items():
        for it in tx:
          cntr[it] += cnt
      
      freq_items = {it for it, cnt in cntr.items() if cnt >= min_count}
      if not freq_items: return None
      
      # tree root
      root = Node()   # the top Null node
      # shortcut to nodes for each item
      head = {it: Head(it, cntr[it]) for it in freq_items}  # {'item': Head}
      # FPTree = root + head
      fptree = cls(root, head)
      
      # merge each string to FPTree
      for tx, cnt in data.items():
        # tag total count to those freq item
        frq_it = {it: head[it].count for it in tx if it in freq_items}
        if frq_it:
          # sort by total count reversely
          it_srt = [v[0] for v in sorted(frq_it.items(), key=lambda p:(p[1], p[0]), reverse=True)]
          fptree.update(it_srt, cnt)
      
      # return the tree
      return fptree
    
    def update(self, tx, cnt):
      def _update(tx, cur):
        if not tx: return
        
        it, rst = tx[0], tx[1:]   # split head/tail
        if it in cur.children:    # add count if already in tree
          cur.children[it].count += cnt
        else:                     # create new branch
          # insert new node to tree
          cur.children[it] = Node(it, cnt, cur)
          # head insert to head
          cur.children[it].next = self.head[it].next
          self.head[it].next = cur.children[it]
        
        _update(rst, cur.children[it])    # recursive merge
      
      _update(tx, self.root)              # merge from root
    
    def prefix_path(self, base_item):
      ret = { }  # {'path': count}
      
      # try each node scattered within headlist
      node = self.head[base_item].next
      while node:
        cur, prf_path = node, [ ]
        while cur.parent:    # collect path in tree from below to top
          prf_path.append(cur.value)
          cur = cur.parent
        if len(prf_path) > 1:             # ignore the base item
          ret[frozenset(prf_path[1:])] = node.count  # count of the base item
        node = node.next
      
      return ret
    
    def find_freqset(self, min_count, ret_freqset):
      '''ret_freqset is an **OUT** parameter'''
      def _find_freqset(fptree, min_count, prefix, ret_freqset):
        if not fptree or not fptree.head: return
        
        # 1-frequent directly from headlist
        #C1 = [v.value for v in sorted(fptree.head.values(), key=lambda p:p.count)]
        C1 = fptree.head.keys()
        
        # for each freqset extend to new freqset
        for base_item in C1:
          nfreqset = prefix.copy()
          nfreqset.add(base_item)
          ret_freqset[len(nfreqset)].append(nfreqset)
          
          # create conditional FPTree for current freqset as prefix
          rdata = fptree.prefix_path(base_item)  # extract ONLY related records as data
          nfptree = FPTree.create(rdata, min_count) # conditional FPTree
          if DEBUG:
            print('base_item=', base_item, 'rdata=', rdata)
            if nfptree: nfptree.display()
            else: print('[Empty FPTree]')
          _find_freqset(nfptree, min_count, nfreqset, ret_freqset)
      
      _find_freqset(self, min_count, set(), ret_freqset)   # from prefix = 'Ø'
  
  min_count = min_support * len(data)
  ddata = {frozenset(tx):1 for tx in data}   # {set: count}
  fptree = FPTree.create(ddata, min_count)
  if DEBUG: fptree.display()
  freqset = defaultdict(list)   # {'k': set}
  fptree.find_freqset(min_count, freqset)
  for k in sorted(freqset.keys()):
    print(f'<< freq-{k} sets: {freqset[k]}')

  return freqset

def find_strong_association(data, freqset, min_confidence):
  # at least |S| >= 2
  freqset = [s for v in freqset.values() for s in v if len(s) > 1]
  rules = [ ]   # (conf, 'X -> Y')
  for S in freqset:
    listS = list(S)    # fix index
    N = len(S)

    # for each subset of s
    itr = 0
    while itr < 2**N-2:  
      itr += 1          # next subset
      X, Y = set(), set()
      for idx, sel in enumerate(bin(itr)[2:].rjust(N, '0')):
        if sel == '1': X.add(listS[idx])
        else: Y.add(listS[idx])

      # calc confidence for 'X -> Y', where Y = S \ XS
      suppX = sum([X.issubset(tx) for tx in data])
      suppS = sum([S.issubset(tx) for tx in data])
      conf = suppS / suppX
      if conf >= min_confidence:
        rules.append((conf, '%s -> %s' % (X, Y)))

  # return all satisfactory
  return rules

def get_data(ds):
  def get_testsample():
    return [{'牛奶','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'},
            {'莳萝','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'},
            {'牛奶','苹果','芸豆','鸡蛋'},
            {'牛奶','独角兽','玉米','芸豆','酸奶'},
            {'玉米','洋葱','芸豆','冰淇淋','鸡蛋'}]
  
  def get_GroceryStore():
    fp = path.join(DATA_DIR, 'GroceryStore.rar')
    rar = rarfile.RarFile(fp, charset='utf8')
    with rar.open('Groceries.csv') as fh:
      data = fh.readall().decode('utf8')
    ret = [ ]
    for line in data.split('\r')[1:]:   # "<id>","{<items>}"
      ln = line[line.find(',')+3:-2]    # ignore '"<id>","{' and '}"'
      ret.append([i.strip() for i in ln.split(',')])
    return ret
  
  def get_UNIX_usage():
    fp = path.join(DATA_DIR, 'UNIX_usage.rar')
    rar = rarfile.RarFile(fp, charset='utf8')
    ret = [ ]
    for i in range(9):
      with rar.open(f'USER{i}/sanitized_all.981115184025') as fh:
        data = fh.readall().decode('utf8')
      PARAM_REGEX = re.compile(r'<\d+>')   
      for line in data.split('**SOF**'):   # tx begins with '**SOF**'
        cmds = line.split('\n')
        for i, cmd in enumerate(cmds):
          if CASE_MERGE_PARAM and PARAM_REGEX.match(cmd):
            #cmds[i] = '<PARAM>'
            cmds[i] = ''
          if cmd in CASE_FILTER:
            cmds[i] = ''
        cmds = list(filter(len, set(cmds)))
        if cmds: ret.append(cmds)
    return ret
  
  fn = locals().get(f'get_{ds}')
  if fn: return fn()
  else: raise

def runtest(case, min_support, BF=False):
  ds = get_data(case)
  if BF:
    BruteForce(ds, min_support)
    print('-'*70)
  Apriori(ds, min_support)
  print('-'*70)
  FP_Growth(ds, min_support)

if __name__ == '__main__':
  if False:
    print('='*70)
    runtest('testsample', 0.5, BF=True)
    print('='*70)
    runtest('GroceryStore', 0.05)
    print('='*70)
    runtest('UNIX_usage', 0.05)
    print('='*70)
  
  if False:
    print('='*70)
    runtest('GroceryStore', 0.05)
    runtest('GroceryStore', 0.03)
    runtest('GroceryStore', 0.01)
    print('='*70)
    runtest('UNIX_usage', 0.05)
    runtest('UNIX_usage', 0.03)
    runtest('UNIX_usage', 0.01)
    print('='*70)

  if True:
    min_support = 0.01
    min_confidence = 0.7
    case = ['GroceryStore', 'UNIX_usage'][1]

    ds = get_data(case)
    freqset = FP_Growth(ds, min_support)
    assoc_rules = find_strong_association(ds, freqset, min_confidence)
    
    print('[Rule]: %d strong rules in total' % len(assoc_rules))
    for conf, rule in sorted(assoc_rules, reverse=True):
      print('%.4f %s' % (conf, rule))