import numpy as np
import pickle
import os

from itertools import permutations
from paths_and_constants import *


    

class digit_permutator:
    
    def get_permute(self, n=0, m=0, pidx=None):

        def make_permutes():
            
            list = []
            for p in permutations([0, 1, 2, 3, 4, 5]):
                list.append(p)
            
            return list
    
        if pidx is not None:
            assert pidx < self.num_perms
            n, m = self.list4[pidx]
            
        perm_dict = make_permutes()
        avoid_pairs = [1, 0, 3, 2, 5, 4]
        p = [[perm_dict[n][i]] for i in range(6)]
        for i_digit in range(1, 12):
            avoid_list = [np.unique(p[avoid_pairs[i]] + [p[i][-1]]) for i in range(6)]
            ok = np.zeros(len(perm_dict), dtype=bool)
            for i_perm, perm in enumerate(perm_dict):
                ok[i_perm] = np.all([not perm[i] in avoid_list[i] for i in range(6)])
            legals = np.argwhere(ok).flatten().astype(int)
            pick = legals[m % len(legals)]
            [p[i].append(perm_dict[pick][i]) for i in range(6)]
        
        return p


    def __init__(self, num_epochs=6, num_digits=12):
        
        assert num_epochs == 6
        assert num_digits == 12
        self.num_epochs = num_epochs
        self.num_digits = num_digits
        
        fname = os.path.join(IDXS_FOLDER, 'params_for_digit_permite')
        with open(fname, 'rb') as fd:
            self.list4 = pickle.load(fd)['m_n_list']
        self.num_perms = len(self.list4)
    
    def __call__(self, data, permid=0):
        
        assert data.shape[0] == self.num_epochs
        assert data.shape[2] == self.num_digits
        
        p = self.get_permute(n=self.list4[permid][0], m=self.list4[permid][1])
        print('---')
        [print(p[i]) for i in range(6)]
        print('---')
        new_data = np.zeros(data.shape)
        for i_digit in range(self.num_digits):
            for i_epoch in range(self.num_epochs):
                #new_data[p[i_epoch][i_digit], :, i_digit] = data[i_epoch, :, i_digit]
                new_data[i_epoch, :, i_digit] = data[p[i_epoch][i_digit], :, i_digit]
        
        return new_data
    
    
    def get_num_perms(self):
        
        return self.num_perms
    


if __name__ == '__main__':
    
    pobj = digit_permutator()
    
    fname = os.path.join(IDXS_FOLDER, 'params_for_digit_permite')
    
    OBSERVE_RESULTS = True
    
    if not OBSERVE_RESULTS:
        try:
            with open(fname, 'rb') as fd:
                list4 = pickle.load(fd)['m_n_list']
                nstart = np.array(list4)[:, 0].max() + 1
        except:
            list4 = []
            nstart = 0
        
        h = np.zeros((100, 6))
        good_mask = np.zeros((720, 100))
        good_perms = []
        for n in range(nstart, 720):
            for m in range(100):
                p = pobj.get_permute(n=n, m=m)
                score = np.min([np.sort(np.bincount(p[i], minlength=6))[3] for i in range(6)])
                if score >= 3:
                    good_perms.append(p)
                    good_mask[n, m] = 1 + (score >= 4).astype(int)
                    #print(n, m,  score)
                    h[m, score] += 1
                    if score == 4:
                        list4.append([n, m])
            if n % 20 == 19:
                print('n =', n)
                for m in range(100):
                    if np.any(h[m] > 0):
                        print(m, '\t', h[m])
                print('combinations with score 3:', h[:, 3].sum())
                print('combinations with score 4:', h[:, 4].sum())
                with open(fname, 'wb') as fd:
                    pickle.dump(dict({'m_n_list': list4}), fd)
    
    
    if OBSERVE_RESULTS:
        
        with open(fname, 'rb') as fd:
            list4 = pickle.load(fd)['m_n_list']
            print('number of good combinations:', len(list4))
        pick_n = 150
        
        for m_n in list4:
            n, m = m_n
            if n == pick_n:
                print('n={}\tm={}'.format(n, m))
                p = pobj.get_permute(n=n, m=m)
                print(np.array(p))
        print('pidx = ', 1000, '\n', np.array(pobj.get_permute(pidx=1000)))
        print('pidx = ', 2000, '\n', np.array(pobj.get_permute(pidx=2000)))
        print('there are {} different permutations'.format(pobj.get_num_perms()))
        print('pidx = ', 1000, '\n', np.array(pobj.get_permute(pidx=3000))) # this one should produce a bug
    
