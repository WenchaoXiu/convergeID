####=====================================================================================================================
####  1. Configure
####=====================================================================================================================

##-----------------
## 1.1 package
##-----------------
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy import stats

from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,roc_curve, auc

%matplotlib inline

##-----------------
## 1.2 utils
##-----------------
def pkl_save(data, file):
    with open(file, "wb") as fp:   #Pickling
        pickle.dump(data, fp)

def pkl_load(file):
    with open(file, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data


####=====================================================================================================================
####  2. Data loading(highly variable genes)
####=====================================================================================================================



####=====================================================================================================================
####  3. SVM classifier
####=====================================================================================================================

##-----------------
## 3.1 SVM model train on before data predict on after data
##-----------------
def SVM_ovo_classifier(label_before, data_before, data_after):
    ## Data prepare
    label_before = label_before.values
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(label_before) # le.classes_ 
    X = data_before.values
    
    ## OVO SVM model
    clf = SVC(C=1.0, cache_size=2000, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X, y)

    ## OVO classifier evaluate
    data_before_DF = clf.decision_function(X) # for ovo
    score_rbf = clf.score(X, y)
    print("The score of rbf is : %f"%score_rbf)

    ## Predict on after data
    data_after = data_after.values
    data_after_pred = clf.predict(data_after)
    data_after_DF = clf.decision_function(data_after) # for ovo

    return le, clf, data_after_pred, data_after_DF, data_before_DF

##-----------------
## 3.2 Get all one vs one model prediction
##-----------------
def get_sample_ovo_class(le, clf, clfDF, k):
#     pred = E85_predict[k]
    n = 0
    clist = []
    class_no = len(clf.classes_)
    for i in range(class_no):
        for j in range(i+1,class_no):
            c1 = clf.classes_[i]
            c2 = clf.classes_[j]  
            if clfDF[k][n]>0:
                clist.append(le.classes_[c1])
            else:
                clist.append(le.classes_[c2])
            n += 1
#     (values,counts) = np.unique(np.array(clist), return_counts=True)
#     ind=np.argmax(counts)
#     most_freq = values[ind]
#     print(most_freq)
#     assert most_freq==pred
    return clist

def get_all_ovo_class(le, clf, clfDF):
    '''From model predicted decision function to each ovo classifier's predicted cell type for each sample.
    Args:
        le: Label encoder of Cell-Type:Label-Category.
        clf: One vs one classifier.
        clfDF: 
    Returns:
        ovo_class: 
    '''
    data_class = []
    for k in range(len(clfDF)):
        sample_class = get_sample_ovo_class(le, clf, clfDF, k)
        data_class.append(sample_class)
    ovo_class = np.array(data_class)
    return ovo_class

def get_pred_label_group(OVOpred, label, OVOnumber):
    '''Combine before and after label; get front-later group counts and ratio
    Args: 
        OVOpred: one vs one classifier prediction on each sample, shape: (sample, nclass*(nclass-1)/2)
        OVOnumber: one vs one classifier number
        label: cell type label of later stage.
    Returns:
        label_pred_cnt: Group counts and ratio in before-after cell type combination
        label_pred: Combination of cell type before and after
    '''
    # original cell barcode
    index = label.index
    # predict true label combine
    predict = pd.Series(OVOpred[:,OVOnumber], index=index, name='cell.pred') # 9 is one of ovo classifiter, 0-65
    label_pred = pd.concat([label, predict], axis=1)
    label_pred_cnt = label_pred.groupby(['cell.label', 'cell.pred']).size().reset_index(name='count')
    label_cnt = label_pred_cnt.groupby('cell.label')['count'].transform('sum')
    label_pred_cnt['ratio'] = label_pred_cnt['count'].div(label_cnt)
    label_pred = label_pred.reset_index().rename(columns={'index':'rawBC'})
    return label_pred_cnt,label_pred


##-----------------
## 3.3 Model mean accuracy
##-----------------
def get_ovo_name():
    ovo_name = []
    label = le.classes_
    N = len(label)
    for i in range(N-1):
        for j in range(i+1, N):
            ovo_name.append('%s@%s'%(label[i],label[j]))
    return ovo_name

def get_ovo_acc(ovo_model_pred, train_label):
    acc_list = []
    ovo_columns = get_ovo_name()
    ovo_pred = pd.DataFrame(ovo_model_pred, columns=ovo_columns)
    ovo_true = train_label
    for k,v in enumerate(ovo_columns):
        c1,c2 = v.split('@')
    #     print(c1, c2)
        idx = (ovo_true==c1) | (ovo_true==c2)
        idx = idx.values
        pred_real = pd.DataFrame({'pred':ovo_pred.loc[idx, v].values, 'real':ovo_true[idx].values})
        val_score = pred_real.apply(lambda x: x['pred']==x['real'], axis=1)
        acc = sum(val_score)/len(val_score)
        acc_list.append(acc)
    return acc_list


####=====================================================================================================================
####  4. Get one OVOclassifier convergent clusters
####=====================================================================================================================
def get_convergent_state(SE_sate_dis, ratio_cutoff=0.1, cnt_cutoff=10, max_fc=2):
    Estate = SE_sate_dis.loc[:,'cell.label'].unique() # all after clusters
    conv_pair_list = []
    # SE_dis_fil = pd.DataFrame()
    SE_dis_fil = []

    for k,v in enumerate(Estate): # each end state SE-state-distribution

        # absolute ratio filter
        Estate = SE_sate_dis.loc[SE_sate_dis.loc[:,'cell.label']==v, :]
        Estate_ratio = Estate.loc[Estate.loc[:,'ratio']>=ratio_cutoff,'ratio'].sort_values(ascending=False)
        Estate_cnt = Estate.loc[Estate.loc[:,'ratio']>=ratio_cutoff,'count'].sort_values(ascending=False)
        Estate_ratio = Estate_ratio.tolist()
        Estate_cnt = Estate_cnt.tolist()
        
        # before clusters ratio filter
        if len(Estate_ratio) > 1:
            ratio_val = Estate_ratio[0]/Estate_ratio[1]
            ratio_bool = ratio_val<max_fc
            cnt_bool = (Estate_cnt[0]>cnt_cutoff) and (Estate_cnt[1]>cnt_cutoff)
            if ratio_bool and cnt_bool:
                front_label = Estate.sort_values('ratio').loc[:, 'cell.pred'].tolist()
                print(Estate.sort_values('ratio'), '\n', v, Estate_ratio, sum(Estate_ratio), '\n')
                
                # convergent cluster
                # SE_dis_fil = pd.concat([SE_dis_fil, Estate.sort_values('ratio')])
                Estate = Estate.sort_values('ratio',ascending=False)
                conv_cluster = [Estate.loc[:,'cell.label'].tolist()[0], 
                    '|'.join(Estate.loc[:,'cell.pred'].tolist()), 
                    '|'.join(Estate.loc[:,'count'].astype('str').tolist()), 
                    ratio_val]
                SE_dis_fil.append(conv_cluster)

                # convergent cluster pair(before_1:after, before_2:after)
                conv_pair = ['%s@%s'%(v, front_label[0]), '%s@%s'%(v, front_label[1])] # 'LaterLabel @ FrontLabel'
                conv_pair_list.append(conv_pair)

    SE_dis_fil = pd.DataFrame(SE_dis_fil, columns=['after', 'before', 'count', 'pct_ratio'])
    SE_dis_fil = SE_dis_fil.sort_values('pct_ratio').reset_index(drop=True)
    return conv_pair_list, SE_dis_fil


####=====================================================================================================================
####  5. Lineage information validation
####=====================================================================================================================
def ham_modify(x, y):
    '''Hamming distance of two indel. uncut-cut:1, different cut:2.
       Distance range: 0~2. 
    '''
    ret = 0
    for i,j in zip(x,y):
        if i!=j:
            if i!='-' and j!='-':
                ret += 2
            else:
                ret += 1
    return ret/len(x)

def get_lineage_dist(SE_state, conv_pair, i):
    '''Get hamming distance of convergent clusters, in same groups and between different groups
    Args:
        SE_state: before-after-rawBC DataFrame.
        conv_pair: before-after celltype pair of two convergent clusters
    Returns:
        dist_in1: Hamming distance of convergent cluster1 
        dist_in2: Hamming distance of convergent cluster2
        dist_in: Inner hamming distance of convergent clusters 
        dist_link: Linker hamming distance of convergent clusters 
    '''
    ## Load lineage infor(indel matrix)
    indel_MTX = pd.read_csv('/mnt/Storage/home/xiuwenchao/Project/cell_tracing_TFpair/2.Cas9/indel_split/indel_resolution/CharacterMatrix/e2a_CharacterMatrix.txt', header=0, sep='\t')
    indel_MTX.loc[:,'cellBC'] = indel_MTX.loc[:,'cellBC'].apply(lambda x:x.split('-')[0])
    
    ## Get cell barcodes of two convergent clusters
    conv_c1 = conv_pair[0].split('@')
    conv_c2 = conv_pair[1].split('@')
    print(conv_c1, conv_c2)
    conv_c1_idx = (SE_state.loc[:,'cell.label']==conv_c1[0]) & (SE_state.loc[:,'cell.pred']==conv_c1[1])
    conv_c2_idx = (SE_state.loc[:,'cell.label']==conv_c2[0]) & (SE_state.loc[:,'cell.pred']==conv_c2[1])
    conv_bc1 = SE_state.loc[conv_c1_idx, 'rawBC']
    conv_bc2 = SE_state.loc[conv_c2_idx, 'rawBC']
    
    ## Get cell indel DataFrame of two convergent clusters
    conv_indel1 = indel_MTX.loc[indel_MTX.loc[:,'cellBC'].isin(conv_bc1),:]
    conv_indel2 = indel_MTX.loc[indel_MTX.loc[:,'cellBC'].isin(conv_bc2),:]
    
    ## Get inner and linker hamming distance of two convergent clusters
    dist_in1 = squareform(pdist(conv_indel1.iloc[:,1:].values, ham_modify))
    dist_in2 = squareform(pdist(conv_indel2.iloc[:,1:].values, ham_modify))
    dist_in = np.concatenate([dist_in1.flatten(), dist_in2.flatten()])
    dist_link = squareform(pdist(pd.concat([conv_indel1, conv_indel2]).iloc[:,1:].values, ham_modify))
    # dist_in1 = pairwise_distances(conv_indel1.replace('-',-1).loc[:,'r0':'r17'], metric = "hamming")
    # dist_in2 = pairwise_distances(conv_indel2.replace('-',-1).loc[:,'r0':'r17'], metric = "hamming")
    # dist_in = np.concatenate([dist_in1.flatten(), dist_in2.flatten()])
    # dist_link = pairwise_distances(pd.concat([conv_indel1, conv_indel2]).replace('-',-1).loc[:,'r0':'r17'], metric = "hamming")
    bound = conv_indel1.shape[0]
    dist_link = dist_link[:bound, bound:]
    dist_link = dist_link.flatten()
    
    ## KS-test for inner and linker distance.
    kstest = stats.ks_2samp(dist_in, dist_link)
    print('dist_in mean: %.4f'%np.mean(dist_in), '\tdist_link mean: %.4f'%np.mean(dist_link))
    print('dist_in median: %.4f'%np.median(dist_in), '\tdist_link median: %.4f'%np.median(dist_link))
    print('Two samples KS-test P-value: ', kstest[1])
    print(kstest)
    
    # boxplot of inner and linker distance
    plt.figure(figsize=(4,4))
    sns.violinplot(data=[dist_in, dist_link], palette={0: "#d53e4f", 1: "#66c2a5"})
    # sns.stripplot(data=[dist_in, dist_link], jitter=True, color=".6")
    plt.xticks(plt.xticks()[0], ['Inner','Linker'])
    plt.ylabel('Modified Hamming distance')
    # plt.title('Lineage distance between cells(p-value:%.2f)'%kstest[1])
    # plt.title(str(conv_c1)+','+str(conv_c2))
    plt.savefig('%s_%s.pdf'%('-'.join(conv_c1), '-'.join(conv_c2)))
    # # statistical annotation
    # x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    # y, h, col = 1. + 0.4, 0.2, 'k'
    # plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
    
    ret = {'dist_in1':dist_in1, 'dist_in2':dist_in2, 'dist_in':dist_in, 
        'dist_link':dist_link, 'dist_mean':[np.mean(dist_in), np.mean(dist_link)], 
        'dist_median':[np.median(dist_in), np.median(dist_link)], 'kstest_pval':kstest[1]}
    return ret



def load_data(path_before, path_after):

    data_before = pd.read_csv(path_before, sep='\t', header=0)
    data_after = pd.read_csv(path_after, sep='\t', header=0)

    ##-----------------
    ## 2.2 label-data
    ##-----------------
    before_label = data_before.loc[:,'cell.label'] 
    after_label = data_after.loc[:,'cell.label']

    data_before = data_before.drop('cell.label', axis=1)
    data_after = data_after.drop('cell.label', axis=1)

    data_before.columns = data_before.columns.str.upper()
    data_after.columns = data_after.columns.str.upper()

    ##-----------------
    ## 2.3 overlap gene
    ##-----------------
    overlap_gene = set.intersection(*map(set,[data_before.columns.tolist(), data_after.columns.tolist()]))

    data_before = data_before.loc[:,overlap_gene]
    data_after = data_after.loc[:,overlap_gene]
    return data_before,before_label,data_after,after_label


def get_conv_pair(data_before, before_label, data_after, after_label):
    E70 = data_before
    E85 = data_after
    E70_label = before_label
    E85_label = after_label

    # SVM OVO prediction
    le, clf, E85_predict, E85_ovoDF, E70_ovoDF = SVM_ovo_classifier(E70_label, E70, E85)
    E85_ovo_predict = get_all_ovo_class(le, clf, E85_ovoDF)

    # top convergent pair in each 
    Nmodel = int(len(le.classes_)*(len(le.classes_)-1)/2)
    E85_conv_pair_list = []
    for i in range(Nmodel):
        E85_SE_dis,E85_SE_state = get_pred_label_group(E85_ovo_predict, E85_label, i) # E85_SE_state is label-predict combine, E85_SE_dis is label-predict group ratio

        # select convergent clusters
        E85_conv_pair,E85_SE_dis_fil = get_convergent_state(E85_SE_dis, 0.1, 10, 4) # filter not convergent clusters, base on OVOclassifier number
        E85_conv_pair_list.append(E85_SE_dis_fil) # all convergent pair for each ovo model,later keep top one in each ovo model

        # calculate ks-test of the smallest pct ratio
        if not E85_SE_dis_fil.empty:
            after_cluster = E85_SE_dis_fil.loc[0,'after'] # 0 means smallest pct ratio
            before_cluster = E85_SE_dis_fil.loc[0,'before'] # 0 means smallest pct ratio
            conv_pair = ['%s@%s'%(after_cluster, before) for before in before_cluster.split('|')] # smallest pct ratio convergent pair
    E85_top_conv_pair = pd.concat(E85_conv_pair_list).loc[0,:].reset_index(drop=True)
    E85_top_conv_pair = E85_top_conv_pair.loc[:, ['after', 'before', 'count', 'pct_ratio']]
    E85_top_conv_pair.columns = ['After_class', 'Before_class_pair', 'Before_pair_count', 'Before_pair_ratio']


    # all convergent pair in each 
    E85_conv_pair_list = []
    for i in range(Nmodel):
        E85_SE_dis,E85_SE_state = get_pred_label_group(E85_ovo_predict, E85_label, i) # E85_SE_state is label-predict combine, E85_SE_dis is label-predict group ratio

        # select convergent clusters
        E85_conv_pair,E85_SE_dis_fil = get_convergent_state(E85_SE_dis, 0.1, 10, 4) # filter not convergent clusters, base on OVOclassifier number
        E85_conv_pair_list.append(E85_SE_dis_fil) # all convergent pair for each ovo model,later keep top one in each ovo model

    E85_all_conv_pair = pd.concat(E85_conv_pair_list).reset_index(drop=True)
    E85_all_conv_pair = E85_all_conv_pair.loc[:, ['after', 'before', 'count', 'pct_ratio']]
    E85_all_conv_pair.columns = ['After_class', 'Before_class_pair', 'Before_pair_count', 'Before_pair_ratio']

    return E85_top_conv_pair,E85_all_conv_pair,Nmodel


def rank_by_conv_strength(conv_pair_candidate, all_conv_pair, after_label, Nmodel):
    E85_top_conv_pair = conv_pair_candidate
    E85_all_conv_pair = all_conv_pair
    E85_label = after_label
    E85_top_conv_pair.loc[:, 'sort_before'] = E85_top_conv_pair.loc[:,'before'].apply(lambda x:'|'.join(sorted(x.split('|'))))
    specificity = E85_all_conv_pair.loc[:,'sort_before'].value_counts()/E85_label.nunique()
    chaos = E85_all_conv_pair.loc[:,'after'].value_counts()/Nmodel
    E85_top_conv_pair.loc[:,'Conv_strength'] = E85_top_conv_pair.apply(lambda x: chaos[x['after']]/specificity[x['sort_before']], axis=1)
    E85_top_conv_pair = E85_top_conv_pair.sort_values('Conv_strength', ascending=False)
    return E85_top_conv_pair






