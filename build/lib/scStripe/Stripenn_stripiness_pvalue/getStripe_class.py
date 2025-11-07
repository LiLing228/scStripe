import numpy as np
import pandas as pd
import math
from .stats import *  
from . import stats  
import time
import random
from joblib import Parallel, delayed
from tqdm import tqdm

def nantozero(nparray):
    where_are_nans = np.isnan(nparray)
    nparray[where_are_nans] = 0
    return nparray

class getStripe:
    def __init__(self, unbalLib, resol, all_chromnames, chromnames, all_chromsizes, chromsizes, core, bfilter):
        self.unbalLib = unbalLib
        self.resol = resol
        self.all_chromnames = all_chromnames
        self.all_chromsizes = all_chromsizes
        self.chromnames = chromnames
        self.chromsizes = chromsizes
        self.core = core
        self.bfilter = bfilter
        self.chromnames2sizes={}
        for i in range(len(self.all_chromnames)):
            self.chromnames2sizes[self.all_chromnames[i]] = self.all_chromsizes[i]

        print("self.chromnames =", self.chromnames)

    def mpmean(self):

        def calc(n):
            with np.errstate(divide='ignore', invalid='ignore'):
                pixels_mean = [0 for x in range(framesize)]
                counts_mean = [0 for x in range(framesize)]
                start = self.resol * n * framesize + 1
                end = self.resol * (n + 1) * framesize
                end2 = self.resol * (n + 2) * framesize
                if end > chrsize:
                    end = chrsize
                if end2 > chrsize:
                    end2 = chrsize
                rows = str(chr) + ":" + str(start) + "-" + str(end)
                cols = str(chr) + ":" + str(start) + "-" + str(end2)
                cfm = self.unbalLib.fetch(rows, cols)
                cfm_rows = cfm.shape[0]
                cfm_cols = cfm.shape[1]
                cfm_max = np.max(cfm)

                for i in range(cfm_rows):
                    for j in range(min(framesize, cfm_cols)):
                        if i + j >= cfm_cols:
                            continue
                        else:
                            count = cfm[i, i + j]
                            if np.isnan(count):
                                count = 0
                            pixels_mean[j] += count
                            counts_mean[j] += 1
                del cfm
            return pixels_mean, counts_mean

       #meantable = [[] for x in range(len(self.chromnames))]
        meantable = {}

        for chridx in range(len(self.chromnames)):
            chr = self.chromnames[chridx]
            print('Processing Chromosome: ' + str(chr))
            chrsize = self.chromnames2sizes[chr]
            rowsize = int(np.ceil(chrsize / self.resol))
            framesize = 400
            nframes = math.ceil(rowsize / framesize)

            with np.errstate(divide='ignore'):
                result = Parallel(n_jobs=self.core)(delayed(calc)(n) for n in tqdm(range(nframes)))

            means = []
            for i in range(len(result[0][1])):
                pixelsum = [result[j][0][i] for j in range(nframes)]
                countsum = [result[j][1][i] for j in range(nframes)]
                pixelsum = sum(pixelsum)
                countsum = sum(countsum)
                meanval = pixelsum/countsum
                means.append(meanval)
            meantable[chr] = means

        return meantable
    


    def getQuantile_slow(self, coolinfo, ChrList, quantile):
        def SplitVal(k,w,r_start,r_end):
            with np.errstate(divide='ignore', invalid='ignore'):
                w_start = k * w
                w_start += r_start
                w_end = (k + 1) * w - 1
                if w_end >= np.floor(CHROMSIZE / nbin):
                    w_end = int(np.floor(CHROMSIZE / nbin))
                w_end += r_start

                w_start = int(w_start)
                w_end = int(w_end)
                r_start = int(r_start)
                r_end = int(r_end)
                temp = self.unbalLib[w_start:w_end, r_start:r_end][self.unbalLib[w_start:w_end, r_start:r_end] > 0]
                return temp

        res = {}
        chrom_size = coolinfo.chromsizes
        chrom_cum_size = np.nancumsum(chrom_size)
        chrom_names = chrom_size.keys()
        nbin = coolinfo.binsize
        # Index for selected chromosomes
        chridx = [c for c in range(len(chrom_names)) if chrom_names[c] in ChrList]
        chridx.sort()
        for ci in chridx:
            CHROM = chrom_names[ci]
            CHROMSIZE = chrom_size[ci]
            L = int(np.ceil(CHROMSIZE / nbin))
            if ci == 0:
                r_start = 0
            else:
                r_start = chrom_cum_size[ci - 1]

            r_end = chrom_cum_size[ci]

            r_start = np.ceil(r_start / nbin) + 1
            r_end = np.ceil(r_end / nbin)

            hs = np.empty(0)

            w = int(np.floor(25000000 / L))

            N_windows = int(np.ceil(L / w))

            W = Parallel(n_jobs=1)(delayed(SplitVal)(ww,w,r_start,r_end) for ww in tqdm(range(N_windows)))
            for i in range(len(W)):
                hs = np.concatenate((hs, W[i]))
            qt = np.quantile(hs, quantile)
            res[CHROM] = qt
            del(hs)
        return res



    def nulldist(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            t_background_start = time.time()

            samplesize = (self.all_chromsizes / np.sum(self.all_chromsizes)) * 1000
            samplesize = np.uint64(samplesize)
            dif = 1000 - np.sum(samplesize)
            notzero = np.where(samplesize != 0)
            chromnames2 = [self.all_chromnames[i] for i in notzero[0]]
            samplesize[0] += dif

            def available_cols(chr):
                with np.errstate(divide='ignore', invalid='ignore'):
                    chr = str(chr)
                    chrsize = self.chromnames2sizes[chr]
                    itera = min(chrsize/self.resol/500, 25)
                    itera = np.uint64(itera)
                    unitsize = np.floor(chrsize/self.resol/itera)
                    unitsize = np.uint64(unitsize)
                    poolsum = 0
                    for it in range(itera):
                        test_region_start = np.uint64(unitsize * self.resol * it+1)
                        test_region_end = np.uint64(unitsize * self.resol * (it+1))
                        if test_region_start > test_region_end:
                            a = test_region_start
                            test_region_start = test_region_end
                            test_region_end = a
                        position = str(chr) + ":" + str(test_region_start) + "-" + str(test_region_end)
                        mat = self.unbalLib.fetch(position, position)
                        mat = nantozero(mat)
                        #mat = np.round(mat, 1)
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        poolsum += (len(matsum) - len(zeroindex[0]))

                return poolsum

            print('3.1. Calculating the number of available columns ...')
            n_available_col = Parallel(n_jobs=self.core)(delayed(available_cols)(chr) for chr in tqdm(chromnames2))

            samplesize = (n_available_col/ np.sum(n_available_col)) * 1000
            samplesize = np.uint64(samplesize)
            dif = 1000 - np.sum(samplesize)
            notzero = np.where(samplesize != 0)
            chromnames2 = [chromnames2[i] for i in notzero[0]]
            samplesize[0] += dif

            def main_null_calc(chr):
                with np.errstate(divide='ignore', invalid='ignore'):
                    background_size = 50000/self.resol
                    background_up = np.floor(background_size / 2)
                    background_down = background_size - background_up
                    background_up = int(background_up)
                    background_down = int(background_down)
                    background_size = int(background_size)
                    chr = str(chr)
                    #c = np.where(chromnames2 == chr)[0]
                    c = chromnames2.index(chr)

                    # Modified in Dec 11 2020
                    ss = samplesize[c]
                    chrsize = self.chromnames2sizes[chr]
                    itera = min(chrsize/self.resol/500, 25)
                    itera = int(itera)
                    unitsize = np.floor(chrsize/self.resol/itera)
                    unitsize = int(unitsize)

                    bgleft_up = np.zeros((400, 0))
                    bgright_up = np.zeros((400, 0))
                    bgleft_down = np.zeros((400, 0))
                    bgright_down = np.zeros((400, 0))
                    n_pool = []

                    for it in range(itera):
                        sss = int(ss/itera)
                        test_region_start1 = int(unitsize * self.resol * it+1)
                        test_region_start0 = test_region_start1 - (400*self.resol)
                        test_region_end1 = int(unitsize * self.resol * (it+1))
                        test_region_end2 = int(unitsize * self.resol * (it+1)+(400*self.resol))
                        if test_region_end2 > chrsize:
                            test_region_end2 = chrsize-1
                        if test_region_end1 > chrsize - 400*self.resol:
                            test_region_end1 = chrsize - 400*self.resol
                        if test_region_start0 <= 1:
                            test_region_start0 = 1
                        test_region_start0 = int(test_region_start0)
                        position1 = str(chr) + ":" + str(test_region_start1) + "-" + str(test_region_end1)
                        position2 = str(chr) + ":" + str(test_region_start0) + "-" + str(test_region_end2)
                        mat = self.unbalLib.fetch(position1, position2)
                        mat = nantozero(mat)
                       # mat = np.round(mat, 1)
                        nrow = mat.shape[0]
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        pool = [x for x in list(range(nrow)) if x not in zeroindex[0].tolist()]
                        pool = [x for x in pool if x > 20 and x < (unitsize - 20)]
                        if it == 0:
                            pool = [x for x in pool if x > 410 and x < mat.shape[1]]
                        n_pool.append(len(pool))
                        if len(pool) == 0:
                            del mat
                        elif len(pool) < sss:
                            randval = random.choices(pool, k=len(pool))
                            tableft_up = np.zeros((400, len(pool)))
                            tabcenter_up = np.zeros((400, len(pool)))
                            tabright_up = np.zeros((400, len(pool)))
                            tableft_down = np.zeros((400, len(pool)))
                            tabcenter_down = np.zeros((400, len(pool)))
                            tabright_down = np.zeros((400, len(pool)))
                            for i in range(len(pool)):
                                x = randval[i]
                                for j in range(0,400):
                                    #det = np.random.choice([0,1])
                                    y_down = x + j
                                    y_up = x - j

                                    #if det == 0:
                                    #    y = x + j
                                    #else:
                                    #    y = x - j
                                    if it > 0 :
                                        y_down = y_down + 400
                                        y_up = y_up + 400

                                    tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                    tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                    tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])

                                    tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                    tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                    tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])


                            bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                            bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                            bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                            bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                            bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                            bgright_down_temp = np.subtract(tabcenter_down, tabright_down)
                            bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                            bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                            del mat
                        else:
                            randval = random.choices(pool, k=sss)
                            tableft_up = np.zeros((400, sss))
                            tabcenter_up = np.zeros((400, sss))
                            tabright_up = np.zeros((400, sss))
                            tableft_down = np.zeros((400, sss))
                            tabcenter_down = np.zeros((400, sss))
                            tabright_down = np.zeros((400, sss))
                            for i in range(sss):
                                x = randval[i]
                                for j in range(0, 400):
                                    y_down = x + j
                                    y_up = x - j

                                    if it > 0:
                                        y_up = y_up + 400
                                        y_down = y_down + 400
                                    tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                    tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                    tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])
                                    tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                    tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                    tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])

                            bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                            bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                            bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                            bgright_down_temp = np.subtract(tabcenter_down, tabright_down)

                            del mat
                            bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                            bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                            bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                            bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                    depl = int(ss - bgleft_up.shape[1])
                    if depl > 0:
                        rich = np.argmax(n_pool)
                        test_region_start1 = int(unitsize * self.resol * rich+1)
                        test_region_start0 = int(test_region_start1 - (400*self.resol))
                        test_region_end1 = int(unitsize * self.resol * (rich+1))
                        test_region_end2 = int(unitsize * self.resol * (rich+1)+(400*self.resol))
                        if test_region_end2 > chrsize:
                            test_region_end2 = chrsize-1
                        if test_region_end1 > chrsize - 400*self.resol:
                            test_region_end1 = chrsize - 400*self.resol
                        if test_region_start0 <= 1:
                            test_region_start0 = 1

                        position1 = str(chr) + ":" + str(test_region_start1) + "-" + str(test_region_end1)
                        position2 = str(chr) + ":" + str(test_region_start0) + "-" + str(test_region_end2)
                        mat = self.unbalLib.fetch(position1, position2)
                        mat = nantozero(mat)
                        #mat = np.round(mat, 1)
                        nrow = mat.shape[0]
                        matsum = np.sum(mat, axis=1)
                        zeroindex = np.where(matsum == 0)
                        pool = [x for x in list(range(nrow)) if x not in zeroindex[0].tolist()]
                        pool = [x for x in pool if x > 20 and x < (unitsize - 20)]
                        randval = random.choices(pool, k=depl)
                        tableft_up = np.zeros((400, depl))
                        tabcenter_up = np.zeros((400, depl))
                        tabright_up = np.zeros((400, depl))
                        tableft_down = np.zeros((400, depl))
                        tabcenter_down = np.zeros((400, depl))
                        tabright_down = np.zeros((400, depl))
                        for i in range(depl):
                            x = randval[i]
                            for j in range(0, 400):
                                y_down = x + j
                                y_up= x - j
                                #det = np.random.choice([0, 1])

                                #if det == 0:
                                #    y = x + j
                                #else:
                                #    y = x - j
                                if it > 0:
                                    y_up = y_up + 400
                                    y_down = y_down+400
                                tableft_up[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_up - background_up):(y_up + background_down)])
                                tabcenter_up[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_up - background_up):(y_up + background_down)])
                                tabright_up[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_up - background_up):(y_up + background_down)])
                                tableft_down[j, i] = np.mean(mat[(x - background_up - background_size):(x - background_up), (y_down - background_up):(y_down + background_down)])
                                tabcenter_down[j, i] = np.mean(mat[(x - background_up):(x + background_down), (y_down - background_up):(y_down + background_down)])
                                tabright_down[j, i] = np.mean(mat[(x + background_down):(x + background_down + background_size), (y_down - background_up):(y_down + background_down)])

                        bgleft_up_temp = np.subtract(tabcenter_up, tableft_up)
                        bgright_up_temp = np.subtract(tabcenter_up, tabright_up)
                        bgleft_down_temp = np.subtract(tabcenter_down, tableft_down)
                        bgright_down_temp = np.subtract(tabcenter_down, tabright_down)

                        del mat
                        bgleft_up = np.column_stack((bgleft_up, bgleft_up_temp))
                        bgright_up = np.column_stack((bgright_up, bgright_up_temp))
                        bgleft_down = np.column_stack((bgleft_down, bgleft_down_temp))
                        bgright_down = np.column_stack((bgright_down, bgright_down_temp))

                return bgleft_up, bgright_up, bgleft_down, bgright_down
            # apply parallel.
            print('3.2. Constituting background ...')
            result = Parallel(n_jobs=self.core)(delayed(main_null_calc)(chr) for chr in tqdm(chromnames2))
            bgleft_up = np.zeros((400,0))
            bgright_up = np.zeros((400,0))
            bgleft_down = np.zeros((400,0))
            bgright_down = np.zeros((400,0))

            for i in range(len(result)):
                if(type(result[i]) == type(None)):
                    continue
                else:
                    blu,bru,bld,brd = result[i]
                    bgleft_up=np.column_stack((bgleft_up,blu))
                    bgright_up=np.column_stack((bgright_up,bru))
                    bgleft_down=np.column_stack((bgleft_down,bld))
                    bgright_down=np.column_stack((bgright_down,brd))

            print('Elapsed time for background estimation: ' + str(np.round((time.time() - t_background_start) / 60, 3)) + ' min')
            return bgleft_up, bgright_up, bgleft_down, bgright_down





    def pvalue(self, bgleft_up, bgright_up, bgleft_down, bgright_down, df):
        background_size = 50000 / self.resol
        background_size = int(background_size)

        np.seterr(divide='ignore', invalid='ignore')
        PVAL = []
        dfsize = len(df)
        with np.errstate(divide='ignore',invalid='ignore'):
            for i in range(dfsize):
                chr = df['chr'].iloc[i]
                chr = str(chr)
                chrLen = self.chromnames2sizes[chr]
                pos1 = df['pos1'].iloc[i]
                pos2 = df['pos2'].iloc[i]
                pos3 = df['pos3'].iloc[i]
                pos4 = df['pos4'].iloc[i]
                leftmost = pos1 - background_size * self.resol
                rightmost = pos2 + background_size * self.resol
                if leftmost < 1:
                    leftmost = 1
                if rightmost > chrLen:
                    rightmost = chrLen
                cd1 = chr + ":" + str(leftmost) + "-" + str(rightmost)
                cd2 = chr + ":" + str(pos3) + "-" + str(pos4)
                mat = self.unbalLib.fetch(cd2, cd1)
                mat_center = mat[:, background_size:(-1*background_size)]
                mat_left = mat[:, :background_size]
                mat_right = mat[:, (-1*background_size):]

                mat_center = nantozero(mat_center)
                mat_left = nantozero(mat_left)
                mat_right = nantozero(mat_right)

                center = np.mean(mat_center, axis=1)
                left = np.mean(mat_left, axis=1)
                right = np.mean(mat_right, axis=1)

                left_diff = np.subtract(center, left)
                right_diff = np.subtract(center, right)

                pvalues = []

                x1 = (pos1 - 1) / self.resol
                x2 = pos2 / self.resol
                y1 = (pos3 - 1) / self.resol
                y2 = pos4 / self.resol

                for j in range(len(center)):
                    if x1 == y1:  # downward stripe
                        difference = j
                        if difference >= 400:
                            difference = 399
                        difference = int(difference)
                        bleft = bgleft_down[difference,:]
                        bright = bgright_down[difference,:]
                    elif x2 == y2:  # upward stripe
                        difference = y2 - y1 - j - 1
                        if difference >= 400:
                            difference = 399
                        difference = int(difference)
                        bleft = bgleft_up[difference,:]
                        bright = bgright_up[difference,:]

                    p1 = len(np.where(bleft >= left_diff[j])[0]) / len(bleft[np.where(~np.isnan(bleft))])
                    p2 = len(np.where(bright >= right_diff[j])[0]) / len(bright[np.where(~np.isnan(bright))])
                    pval = max(p1, p2)
                    if pval == 0:
                        pval = 1/len(bleft)
                    pvalues.append(pval)
                PVAL.append(np.median(pvalues))
        return PVAL




    def scoringstripes(self, df, expecVal, mask='0'):
        background_size = 50000/self.resol
        background_size = int(background_size)
        df['chr'] = df['chr'].astype(str)  

        if mask != '0':
            mask = mask.split(':')
            mask_chr = mask[0]
            mask_start = int(mask[1].split('-')[0])
            mask_end = int(mask[1].split('-')[1])
            mask_x_start = int(mask_start / self.resol)
            mask_x_end = int(mask_end / self.resol)

        def masking(matrix ,mask_index_start, mask_index_end, start_index, end_index, direc):
            matrix = matrix.astype('float')
            relative_mask_index_start = mask_index_start - start_index
            mask_size = mask_index_end - mask_index_start
            relative_mask_index_end = relative_mask_index_start + mask_size
            L = end_index - start_index +1
            idx = [x for x in range(relative_mask_index_start, relative_mask_index_end+1) if x in range(L)]
            if len(idx) == 0:
                return matrix
            else:
                if direc == 1:
                    for x in range(matrix.shape[0]):
                        for y in idx:
                            matrix[x,y] = np.nan
                if direc == 2:
                    for x in idx:
                        for y in range(matrix.shape[1]):
                            matrix[x,y] = np.nan

            return matrix

        def expecMatrix(exval, x_start_index, x_end_index, y_start_index, y_end_index):
            x_seq = range(x_start_index, x_end_index)
            y_seq = range(y_start_index, y_end_index)
            x_seq = np.array(x_seq)
            y_seq = np.array(y_seq)

            index_matrix = [[abs(x_seq - y_seq[i])] for i in range(len(y_seq))]
            index_matrix = np.array(index_matrix)
            expec_matrix = np.empty(shape = (index_matrix.shape[0],index_matrix.shape[2]))

            for x in range(len(x_seq)):
                for y in range(len(y_seq)):
                    idx = index_matrix[y][0][x]
                    if idx >= 400:
                        idx = 399
                    expec_matrix[y,x] = exval[idx]

            del index_matrix
            return expec_matrix

        def iterate_idx(i, is_mask, exval):
            def safe_region(chrom, start, end):
                if start > end:
                    start, end = end, start
                return f"{chrom}:{start}-{end}"

            with np.errstate(divide='ignore', invalid='ignore'):
                xs = df['pos1'].iloc[i]
                xe = df['pos2'].iloc[i]
                ys = df['pos3'].iloc[i]
                ye = df['pos4'].iloc[i]

                x_start_index = int(xs / self.resol)
                x_end_index = int(xe / self.resol)
                y_start_index = int(ys / self.resol)
                y_end_index = int(ye / self.resol)

                leftmost = x_start_index - background_size
                rightmost = x_end_index + background_size
                if leftmost < 1:
                    leftmost = 1
                if rightmost >= chrom_bin_size:
                    rightmost = chrom_bin_size - 1

                print(f"[INFO] Stripe idx={i} chr={c}, xs={xs}, xe={xe}, ys={ys}, ye={ye}, x_start_index={x_start_index}, x_end_index={x_end_index}, background_size={background_size}")

                x_coord = safe_region(c, int(xs), int(xe))
                y_coord = safe_region(c, int(ys), int(ye))
                center_obs = self.unbalLib.fetch(y_coord, x_coord)
                center_exp = expecMatrix(exval, x_start_index, x_end_index, y_start_index, y_end_index)
                center_exp += 1e-8
                center = np.divide(center_obs, center_exp)

                if int(leftmost * self.resol) > int(x_start_index * self.resol):
                    print(f"[DEBUG] left > right (left flank): chr={c}, left={int(leftmost * self.resol)}, right={int(x_start_index * self.resol)}, stripe_idx={i}")
                x_coord = safe_region(c, int(leftmost * self.resol), int(x_start_index * self.resol))
                left_obs = self.unbalLib.fetch(y_coord, x_coord)
                left_exp = expecMatrix(exval, leftmost, x_start_index, y_start_index, y_end_index)
                left_exp += 1e-8
                left = np.divide(left_obs, left_exp)

                if int(x_end_index * self.resol) > int(rightmost * self.resol):
                    print(f"[DEBUG] left > right (right flank): chr={c}, left={int(x_end_index * self.resol)}, right={int(rightmost * self.resol)}, stripe_idx={i}")
                x_coord = safe_region(c, int(x_end_index * self.resol), int(rightmost * self.resol))
                right_obs = self.unbalLib.fetch(y_coord, x_coord)
                right_exp = expecMatrix(exval, x_end_index + 1, rightmost + 1, y_start_index, y_end_index)
                right_exp += 1e-8
                right = np.divide(right_obs, right_exp)

                center = nantozero(center)
                left = nantozero(left)
                right = nantozero(right)

                centerm = np.mean(center, axis=1)
                leftm = np.mean(left, axis=1)
                rightm = np.mean(right, axis=1)

                centerTotal = np.sum(center[np.where(~np.isnan(center))[0]])
                centerMean = np.mean(center[np.where(~np.isnan(center))[0]])

                g_xl = stats.elementwise_product_sum(K_xl, leftm, centerm)
                g_xr = stats.elementwise_product_sum(K_xr, centerm, rightm)
                g_y = stats.elementwise_product_sum(K_y, leftm, centerm, rightm)
                g_x = np.minimum(g_xl, g_xr)
                diff = [a - b for a, b in zip(g_x, g_y)]
                diff = [x for x in diff if x >= 0 or x < 0]
                avgdiff = np.mean(diff)
                g = np.nanmedian(centerm) * avgdiff
                g = float(g)
                return i, g, centerMean, centerTotal


        # Scoring
        ### Sobel-like operators
        listg = [0 for x in range(df.shape[0])]
        listMean = [0 for x in range(df.shape[0])]
        listTotal = [0 for x in range(df.shape[0])]

        K_xl = np.array([[-1, 1], [-2, 2], [-1, 1]])
        K_xr = np.array([[1, -1], [2, -2], [1, -1]])
        K_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        chrset = list(set(df['chr']))
        for c in chrset:
            is_mask = False
            if mask!='0':
                is_mask = ( mask_chr == c )

            idx = np.where(df['chr'] == c)[0].tolist()
            chrom_idx = self.chromnames.index(c)
            chrom_bin_size = np.ceil(self.chromsizes[chrom_idx] / self.resol)
            chrom_bin_size = int(chrom_bin_size)
            exval = expecVal[c]

            result = Parallel(n_jobs=self.core)(delayed(iterate_idx)(i,is_mask,exval) for i in tqdm(idx))
            for r in range(len(result)):
                i,g,cM,cT = result[r]
                listg[i] = g
                listMean[i] = cM
                listTotal[i] = cT
        return listg,listMean,listTotal


    def extract(self, res, bgleft_up, bgright_up, bgleft_down, bgright_down):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Stripe filtering and scoring
            p = self.pvalue(bgleft_up, bgright_up, bgleft_down, bgright_down, res)
            res = res.assign(pvalue=pd.Series(p))

        return res



