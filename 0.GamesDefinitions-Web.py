#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as ospath

from scipy.spatial import distance


# In[7]:


preurl = 'https://raw.githubusercontent.com/moskvo/allocation-games/gh-pages/SeriousGames_2020Sept_'

Alldata = pd.read_csv(preurl+'Data.csv',sep=';',decimal=',').sort_values(by=['Game','Time','GrSubject'])
Alldata['Util'] = Alldata['Gain'] + Alldata['Penalty']
Games = pd.read_csv(preurl+'Games.csv',sep=';',decimal=',')
(Alldata.columns,Games.columns)


# In[8]:


Players = pd.read_csv(preurl+'Players.csv',sep=';',decimal=',')
GamesPlayers = pd.read_csv(preurl+'GamesOfPlayers.csv',sep=';',decimal=',')
Questionary = pd.read_csv(preurl+'Questionary.csv',sep=';',decimal=',')


# ### КП

# In[4]:


def aheadCB(diter,eps,metric='cityblock'):
    """ return next CB start and stop and remaining iterator """
    from scipy.spatial import distance
    from itertools import chain as mchain
    start,vals = next(diter)
    vals = [ vals ]
    
    prev = start
    for i,v in diter:
        if max(distance.cdist(vals,[v],metric)).item(0) > eps :
            return ((start,prev),mchain([(i,v)],diter))
        else:
            prev = i
            vals.append( v )
    return ((start,prev),diter)

def getCB(data,eps=0,metric='cityblock'): # colon data[0] must be 1,2,3,... to properly work of data[end-1] 
    from itertools import chain as mchain
    dend = data.index[-1]
    (start,end),diter = aheadCB(zip(data.index.values,data.values),eps,metric=metric)
    if start != end :
        cblist = [(start,end)]
        diter = mchain([(end,data.loc[end].values)],diter)
    else :
        cblist = []

    while end != dend :
        ((start,end),diter) = aheadCB(diter,eps)
        if start != end :
            cblist.append((start,end))
            diter = mchain([(end,data.loc[end].values)],diter)
    return cblist


# In[5]:


def OnlySelfBidsOfGame(game): # s2 = sii
    shiftgame = game.copy()
    n = game.columns.get_loc('s1') - 1
    shiftgame.loc[:,'s2']=shiftgame.apply(lambda x: x[n+x['GrSubject']], axis=1)
    return shiftgame

def getCBwithEps(gamesdata, eps=0,dim=1,metric='cityblock'):    
    cols = ['s2'] if dim == 1 else ['s1','s2','s3']
    gr = gamesdata.loc[:,['Game','Time','GrSubject']+cols].groupby(['GrSubject','Game'])
    
    lst = []
    for (s,g),data in gr:
        data = data.set_index('Time')
        glen = data.index.max()
        if dim == 1:
            cb = getCB(data[cols],eps,metric=metric)
        else:
            cb = getCB(data[cols],eps,metric=metric)
        data = pd.DataFrame(cb,columns=['ts','te'])
        data['subject'] = s
        data['game'] = g
        data['gamelength'] = glen
        lst.append(data)
    return pd.concat(lst).reset_index(drop=True)
    
def getCBofAllGames(gamesdata, maxeps=10, dim=1,metric='cityblock'):
    if dim == 1:
        gr = gamesdata.loc[:,['Game','Time','GrSubject','s2']].groupby(['GrSubject','Game'])
    else:
        gr = gamesdata.loc[:,['Game','Time','GrSubject','s1','s2','s3']].groupby(['GrSubject','Game'])
    cbCounts, stepsCounts = [],[]
    for eps in range(maxeps+1):
        lst = []
        for (s,g),data in gr:
            data = data.set_index('Time')
            if dim == 1:
                cb = getCB(data[['s2']],eps,metric=metric)
            else:
                cb = getCB(data[['s1','s2','s3']],eps,metric=metric)
            data = pd.DataFrame(cb,columns=['ts','te'])
            data['subject'] = s
            data['game'] = g
            lst.append(data)
        cb = pd.concat(lst).reset_index(drop=True)
        cbCounts.append(cb.shape[0])
        stepsCounts.append((cb['te']-cb['ts']).sum())
    Counts = pd.DataFrame({'CB Count':cbCounts,'Steps Count':stepsCounts})
    Counts.index.name = 'eps'
    return Counts

def DataWoCB(gamedata, CBdata, cbeps=0):
    """Game data without constant steps: it takes first steps of an each cb and drops out each other steps from game data."""
    if not CBdata is None:
        CBdata = getCBwithEps(gamedata,cbeps)
    wd = gamedata.copy()
    for row in CBdata.itertuples(False):
        wd = wd[(wd['Game']!=row.game) | (wd['GrSubject']!=row.subject) | (wd['Time']<=row.ts) | (wd['Time']>row.te)]
    return wd

def DataMarkedCB(gamedata, cb, columnname='iscb'):
    """Game data with constant steps: it marks steps of game with "it's CB of the player i" or not. V.1.2"""
    df = gamedata.copy()
    res = []
    for row in gamedata.itertuples(False):
        #t = cb[(cb['game']==row.Game)]
        #t = t[t['subject']==row.GrSubject]
        #t = t[t['ts']<row.Time]
        #rowcb = t[row.Time<=t['te']]
        rowcb = cb[(cb['game']==row.Game) & (cb['subject']==row.GrSubject) & (cb['ts']<row.Time) & (row.Time<=cb['te'])]
        cbflag = 1.0 if rowcb.shape[0] > 0 else 0.0           
        res.append( cbflag )
    df.insert(df.shape[1],columnname,res)
    return df


# ## Добавляем столбцы про торги Нэша

# ### Функции расчёта выигрышей в игре

# In[ ]:


class Game:
    import numpy as np
    def __init__(self,R,s0,types):
        self.R = R
        self.s0 = np.array(s0)
        self.n = len(s0)
        self.types = np.array(types)
    def u(self,x):
        return self.np.sqrt(self.types+x)
game = Game( 115,(115/3,115/3,115/3), (1,9,25))

class Mechanism:
    def __init__(self,game,params,xfunc,tfunc,sfunc):
        self.game = game
        for k,v in params.items():
            setattr(self,k,v)
        self.xfunc = xfunc
        self.tfunc = tfunc
        self.sfunc = sfunc
        
    def get_s(self,pandas_df):
        return self.sfunc(pandas_df)
    def x(self,s):
        return self.xfunc(s,self.game)
    def t(self,s):
        return self.tfunc(s,self.game,self)
    def f(self,s):
        return game.u(self.x(s))-self.t(s)

YHMechanism = Mechanism(game,
                        {'beta':0.0005}, 
                        lambda s,g: s*g.R/s.sum(),
                        lambda s,g,m: m.beta*s*(np.repeat(s.sum(),3)-s),
                        lambda df: df['s2'].to_numpy()
                       )

class GLClass:
    from scipy.spatial import distance
    def __init__(self):
        self.glx = lambda s,g: s.sum(axis=0) / g.n
    def glt(self,s,g,m):
        p = m.beta * self.distance.cdist([self.glx(s,g)],s,'sqeuclidean')[0,:]
        pmean = m.alfa * p.mean()
        return p - pmean
GL = GLClass()
    
GLMechanism = Mechanism(game,{'beta':0.0005,'alfa':1},
                        GL.glx,
                        GL.glt,
                        lambda df: df[['s1','s2','s3']].to_numpy()
                        )

class ADMMMechanismClass(Mechanism):
    def get_s(self,pandas_df):
        return pandas_df.s2.to_numpy()
    def x(self,s):
        return s
    def t(self,s,prev_s,prev_xm,prev_y):
        return self.beta * (s - prev_s + prev_xm + prev_y)**2
    def f(self,s,prev_s,prev_xm,prev_y):
        return  self.game.u(self.x(s)) - self.t(s,prev_s,prev_xm,prev_y)


# In[48]:


print("BR")

# s = np.array([[100.0, 4.5, 10.5], [49.0, 41.0, 25.0], [38.25, 27.75, 49.0 ]])
# xmi = pd.DataFrame(s[1:3,:]).mean()
# maxf = -10000
# maxs = 0
# # find maximum for 1 player
# for s11 in np.arange(xmi[0],200,0.01):
#     d = (s11 - xmi[0]) / 2
#     s12 = xmi[1] - d
#     s13 = xmi[2] - d
#     ts = np.array([[s11,s12,s13]])
#     s1 = np.concatenate(( ts, s[1:3] ))
#     #print(s1)
#     f = GLMechanism.f(s1)[0][0]
#     if ( f > maxf ):
#         maxf = f
#         maxs = s11

    


# In[18]:


print(YHMechanism.game.s0)

f0 = YHMechanism.game.u([0,0,0])
print(f0)


# In[27]:


t = np.array([1,2,3])
t2 = np.array([0,1,2])
t3 = np.array([[0,1,2],[1,2,3],[2,3,4]])


# In[44]:


GLMechanism.x(t3)


# In[45]:


GLMechanism.t(t3)


# * **Ui>Uprev**           : $ U(s^i(t),d) > U(s(t-1),d),$
# * **Uall>Uprev**         :  Uall_more_Uprev = U1_more_Uprev + U2_more_Uprev + U3_more_Uprev 
# * **Unew>Uprev**         : $ U(s(t),d) > U(s(t-1),d) $ 
# * **Uloci>Ulocprev** : $ U_{loc}(s^i(t),d) > U_{loc}(s(t-1),d)$ 
# * **Ulocnew>Ulocprev**  : $ U_{loc}(s(t),d) > U_{loc}(s(t-1),d)$ 
# * **Fnew>Fprev**         : $ g(s(t)) > g(s(t-1)) $ 
# * **Fi>Fprev**       : $ g(s^i(t)) > g(s(t-1)) $ 
# * **Fii>Fiprev**       : $ g_i(s^i(t)) > g_i(s(t-1)) $ - рациональность поступка игрока? 
# 
# where $s^i(t) = (s_i(t),s_{-i}(t-1)) $ and $ d = (0,0,0)$
# where g = utility - transfer

# In[35]:


class NashBargaining:
    def __init__(self,f0,prec=0.000001):
        self.f0 = f0
        self.prec = prec
    def Unash(self,f,f0):
        df = f-f0
        return (np.fabs(df)).prod() * np.min(np.sign(df)) # если хоть по одной ЦФ было уменьшение, то отрицательна

    # Функция получения исходных данных для Нэш-торгов
    def NashData(self,Data,Mechanism):
        import itertools
        f0 = self.f0
        prec = self.prec
        res = []
        GroupedData = Data.groupby(['Game','Time'])
        for name,group in GroupedData:
            group = group.sort_values(by='GrSubject')
            if name[1] == 1 : # второй элемент - время, т.е. если перешли к новой игре
                prevg = group
                continue # go to second step

            # предыдущие выигрыши
            #fprev = prevg['Gain'].values
            sprev = Mechanism.get_s(prevg)
            Fprev = Mechanism.f(sprev)
            Uprev = self.Unash(Fprev,f0)
            s = Mechanism.get_s(group)
            s1, s2, s3 = sprev.copy(), sprev.copy(), sprev.copy()
            # действия игроков "по одному", их стремление
            s1[0] = s[0]
            s2[1] = s[1]
            s3[2] = s[2]

            F1 = Mechanism.f(s1)
            F1_more_Fprev = 1.0 if (F1>Fprev+prec).all() else 0.0 # s1 - strong Pareto
            F1_notless_Fprev = 1.0 if (F1>Fprev-prec).all() else 0.0 # s1 - weak Pareto
            F11_more_F1prev = 1.0 if (F1[0]>Fprev[0]+prec) else 0.0 # gain of 1 player have increased with s1 bids
            U1 = self.Unash(F1,f0)
            U1_more_Uprev = 1.0 if U1>Uprev+prec else 0.0

            F2 = Mechanism.f(s2)
            F2_more_Fprev = 1.0 if (F2>Fprev+prec).all() else 0.0 # s2 - strong Pareto
            F2_notless_Fprev = 1.0 if (F2>Fprev-prec).all() else 0.0 # s2 - weak Pareto
            F22_more_F2prev = 1.0 if (F2[1]>Fprev[1]+prec) else 0.0
            U2 = self.Unash(F2,f0)
            U2_more_Uprev = 1.0 if U2>Uprev+prec else 0.0

            F3 = Mechanism.f(s3)
            F3_more_Fprev = 1.0 if (F3>Fprev+prec).all() else 0.0 # s3 - strong Pareto
            F3_notless_Fprev = 1.0 if (F3>Fprev-prec).all() else 0.0 # s3 - weak Pareto
            F33_more_F3prev = 1.0 if (F3[2]>Fprev[2]+prec) else 0.0
            U3 = self.Unash(F3,f0)
            U3_more_Uprev = 1.0 if U3>Uprev+prec else 0.0

            Uall_more_Uprev = U1_more_Uprev + U2_more_Uprev + U3_more_Uprev

            Fnew = Mechanism.f(s)
            Fnew_more_Fprev = 1.0 if (Fnew>Fprev+prec).all() else 0.0 # s - strong Pareto
            Fnew_notless_Fprev = 1.0 if (Fnew>Fprev-prec).all() else 0.0 # s - weak Pareto
            Unew = self.Unash(Fnew,f0)
            Unew_more_Uprev = 1.0 if Unew>Uprev+prec else 0.0

            Ulocnew = self.Unash(Fnew,Fprev)
            Ulocnew_more_Ulocprev = -1.0
            Uloc1_more_Ulocprev = -1.0
            Uloc2_more_Ulocprev = -1.0
            Uloc3_more_Ulocprev = -1.0
            Uloc1 = -1.0
            Uloc2 = -1.0
            Uloc3 = -1.0
            if name[1]>2 :
                Ulocprev = res[-1][-2]
                Uloc1 = self.Unash(F1,Fprev)
                Uloc1_more_Ulocprev = 1.0 if Uloc1 > Ulocprev+prec else 0.0
                Uloc2 = self.Unash(F2,Fprev)
                Uloc2_more_Ulocprev = 1.0 if Uloc2 > Ulocprev+prec else 0.0
                Uloc3 = self.Unash(F3,Fprev)
                Uloc3_more_Ulocprev = 1.0 if Uloc3 > Ulocprev+prec else 0.0
                Ulocnew_more_Ulocprev = 1.0 if Ulocnew > Ulocprev+prec else 0.0

            res.append( [a for a in itertools.chain(name,[U1_more_Uprev, U2_more_Uprev, U3_more_Uprev,
                                                          Uloc1_more_Ulocprev, Uloc2_more_Ulocprev, Uloc3_more_Ulocprev,
                                                          Uall_more_Uprev, Unew_more_Uprev,
                                                          Fnew_more_Fprev, F1_more_Fprev, F11_more_F1prev, F2_more_Fprev, F22_more_F2prev, F3_more_Fprev, F33_more_F3prev,
                                                          Fnew_notless_Fprev, F1_notless_Fprev, F2_notless_Fprev, F3_notless_Fprev,
                                                          Uprev, U1,U2,U3, Uloc1,Uloc2,Uloc3, Unew,Ulocnew, Ulocnew_more_Ulocprev])] )
            prevg = group
        data_a = pd.DataFrame(np.vstack(res),
                          columns=['Game','Time','U1>Uprev','U2>Uprev','U3>Uprev','Uloc1>Ulocprev','Uloc2>Ulocprev','Uloc3>Ulocprev',
                                   'Uall>Uprev','Unew>Uprev',
                                   'Fnew>Fprev','F1>Fprev', 'F11>F1prev', 'F2>Fprev', 'F22>F2prev', 'F3>Fprev', 'F33>F3prev',
                                   'Fnew>=Fprev','F1>=Fprev', 'F2>=Fprev', 'F3>=Fprev',
                                   'Uprev','U1','U2','U3','Uloc1','Uloc2','Uloc3','Unew','Ulocnew','Ulocnew>Ulocprev'])

        return data_a

    # Функция получения исходных данных для Нэш-торгов по механизму ADMM
    # Data sorted by [Game,Time,GrSubject] increasingly
    def NashDataADMM(self,Data,Mechanism):
        import itertools
        Rmean = Mechanism.game.R / Mechanism.game.n
        f0 = self.f0
        prec = self.prec
        res = []
        GroupedData = Data.groupby(['Game','Time'])
        for name,group in GroupedData:
            group = group.sort_values(by='GrSubject')
            if name[1] == 1 : # второй элемент - время, т.е. если перешли к новой игре
                prevg = group
                prev_y = 0 + np.mean(group.x) - Rmean # dual variable in ADMM, 0 is base value of y
                #sprevprev = np.array([rmean])
                continue # go to second step

            # предыдущие выигрыши
            Fprev = prevg['Gain'].values
            sprev = Mechanism.get_s(prevg)
            #Fprev = Mechanism.f(sprev)
            Uprev = self.Unash(Fprev,f0)
            s = Mechanism.get_s(group)            
            s1, s2, s3 = sprev.copy(), sprev.copy(), sprev.copy()
            # действия игроков "по одному", их стремление
            s1[0] = s[0]
            s2[1] = s[1]
            s3[2] = s[2]

            prev_xm = np.mean(prevg.x) - Rmean
            # check
            tFcalc = Mechanism.f(s,sprev, prev_xm, prev_y)
            tFhist = group.Gain
            if ((tFcalc - tFhist) > prec).any():
                print( 'Not equal for game,time=', name, ' . Calculated: ', tFcalc, ', saved: ', tFhist )

            F1 = Mechanism.f(s1, sprev, prev_xm, prev_y)
            F1_more_Fprev = 1.0 if (F1>Fprev+prec).all() else 0.0
            F1_notless_Fprev = 1.0 if (F1>Fprev-prec).all() else 0.0
            F11_more_F1prev = 1.0 if (F1[0]>Fprev[0]+prec) else 0.0
            U1 = self.Unash(F1,f0)
            U1_more_Uprev = 1.0 if U1>Uprev+prec else 0.0

            F2 = Mechanism.f(s2, sprev, prev_xm, prev_y)
            F2_more_Fprev = 1.0 if (F2>Fprev+prec).all() else 0.0
            F2_notless_Fprev = 1.0 if (F2>Fprev-prec).all() else 0.0
            F22_more_F2prev = 1.0 if (F2[1]>Fprev[1]+prec) else 0.0
            U2 = self.Unash(F2,f0)
            U2_more_Uprev = 1.0 if U2>Uprev+prec else 0.0

            F3 = Mechanism.f(s3, sprev, prev_xm, prev_y)
            F3_more_Fprev = 1.0 if (F3>Fprev+prec).all() else 0.0
            F3_notless_Fprev = 1.0 if (F3>Fprev-prec).all() else 0.0
            F33_more_F3prev = 1.0 if (F3[2]>Fprev[2]+prec) else 0.0
            U3 = self.Unash(F3,f0)
            U3_more_Uprev = 1.0 if U3>Uprev+prec else 0.0

            Uall_more_Uprev = U1_more_Uprev + U2_more_Uprev + U3_more_Uprev

            Fnew = Mechanism.f(s, sprev, prev_xm, prev_y)
            Fnew_more_Fprev = 1.0 if (Fnew>Fprev+prec).all() else 0.0
            Fnew_notless_Fprev = 1.0 if (Fnew>Fprev-prec).all() else 0.0
            Unew = self.Unash(Fnew,f0)
            Unew_more_Uprev = 1.0 if Unew>Uprev+prec else 0.0

            Ulocnew = self.Unash(Fnew,Fprev)
            Ulocnew_more_Ulocprev = -1.0
            Uloc1_more_Ulocprev = -1.0
            Uloc2_more_Ulocprev = -1.0
            Uloc3_more_Ulocprev = -1.0
            Uloc1 = -1.0
            Uloc2 = -1.0
            Uloc3 = -1.0
            if name[1]>2 :
                Ulocprev = res[-1][-2]
                Uloc1 = self.Unash(F1,Fprev)
                Uloc1_more_Ulocprev = 1.0 if Uloc1 > Ulocprev+prec else 0.0
                Uloc2 = self.Unash(F2,Fprev)
                Uloc2_more_Ulocprev = 1.0 if Uloc2 > Ulocprev+prec else 0.0
                Uloc3 = self.Unash(F3,Fprev)
                Uloc3_more_Ulocprev = 1.0 if Uloc3 > Ulocprev+prec else 0.0
                Ulocnew_more_Ulocprev = 1.0 if Ulocnew > Ulocprev+prec else 0.0

            res.append( [a for a in itertools.chain(name,[U1_more_Uprev, U2_more_Uprev, U3_more_Uprev,
                                                          Uloc1_more_Ulocprev, Uloc2_more_Ulocprev, Uloc3_more_Ulocprev,
                                                          Uall_more_Uprev, Unew_more_Uprev,
                                                          Fnew_more_Fprev, F1_more_Fprev, F11_more_F1prev, F2_more_Fprev, F22_more_F2prev, F3_more_Fprev, F33_more_F3prev,
                                                          Fnew_notless_Fprev, F1_notless_Fprev, F2_notless_Fprev, F3_notless_Fprev,
                                                          Uprev, U1,U2,U3, Uloc1,Uloc2,Uloc3, Unew,Ulocnew, Ulocnew_more_Ulocprev])] )
            prevg = group
            prev_y = prev_y + np.mean(group.x) - Rmean
        data_a = pd.DataFrame(np.vstack(res),
                          columns=['Game','Time','U1>Uprev','U2>Uprev','U3>Uprev','Uloc1>Ulocprev','Uloc2>Ulocprev','Uloc3>Ulocprev',
                                   'Uall>Uprev','Unew>Uprev',
                                   'Fnew>Fprev','F1>Fprev', 'F11>F1prev', 'F2>Fprev', 'F22>F2prev', 'F3>Fprev', 'F33>F3prev',
                                   'Fnew>=Fprev','F1>=Fprev', 'F2>=Fprev', 'F3>=Fprev',
                                   'Uprev','U1','U2','U3','Uloc1','Uloc2','Uloc3','Unew','Ulocnew','Ulocnew>Ulocprev'])

        return data_a


# In[6]:


def NashData_bySubjects(Mechanism, GameData, d0, admmflag=False):
    def nbcols(i):
        return ['Game','Time',f'U{i}>Uprev',f'Uloc{i}>Ulocprev', 'Uall>Uprev','Unew>Uprev','Fnew>Fprev','Fnew>=Fprev',f'F{i}>Fprev',f'F{i}>=Fprev',f'F{i}{i}>F{i}prev',f'U{i}',f'Uloc{i}','Unew','Uprev','Ulocnew','Ulocnew>Ulocprev']

    f0 = Mechanism.game.u(d0)
    NB = NashBargaining(f0)

    nbdata = NB.NashDataADMM(GameData,Mechanism) if admmflag else NB.NashData(GameData,Mechanism)
    # print(nbdata.head())

    nb1 = nbdata[nbcols(1)].copy()
    nb1.columns = nbcols('i')
    nb1['GrSubject'] = 1

    nb2 = nbdata[nbcols(2)].copy()
    nb2.columns = nbcols('i')
    nb2['GrSubject'] = 2

    nb3 = nbdata[nbcols(3)].copy()
    nb3.columns = nbcols('i')
    nb3['GrSubject'] = 3

    selfnbdata = pd.concat([nb1,nb2,nb3])
    # print(selfnbdata.head())
    return selfnbdata


# In[ ]:




