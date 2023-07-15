"""
Università degli Studi di Bari Aldo Moro

@authors: N. Fanizzi & C. Riefolo (c.riefolo2@studenti.uniba.it)
"""

import types
import numpy as np
from random import seed
import pandas as pd
from owlready2 import onto_path, get_ontology, sync_reasoner_pellet, Thing, Not, reasoning  # , reasoning, IRIS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Absolute path
onto_path.append("onto")

#onto = get_ontology("lubm.owl")
#onto = get_ontology("financial-abbrev.owl")
#onto = get_ontology("NTNames.owl")
onto = get_ontology("KRKZEROONE.owl")

reasoning.JAVA_MEMORY = 4000

SEED = 42  # 1
seed(SEED)
print('Seed:', SEED)

N_SPLITS = 10
TEST_PORTION = 0.3

UNLABEL = -1

onto.load(reload=True)
print("Base IRI: ", onto.base_iri)

classes = list(onto.classes())
print('#Classes: %d' % (len(classes)))

for i, a_class in enumerate(classes):
    print('%5d: %s' % (i, a_class))

# print('\n\t# disj. classes: %d' % len(list(onto.disjoint_classes())))
# print('\t%s\n' % list(onto.disjoint_classes()))

oprops = list(onto.object_properties())
print('#Obj-props: %d ' % (len(oprops)))

for j, obj_property in enumerate(oprops):
    print('%5d: %s (%s >> %s)' % (j, obj_property, obj_property.domain, obj_property.range))

dprops = list(onto.data_properties())
print('#Data-props: %d ' % (len(dprops)))

for j, d_property in enumerate(dprops):
    print('%5d: %s (%s >> %s)' % (j, d_property, d_property.domain, d_property.range))

# Individuals
inds = list(onto.individuals())
print('#Individuals ASSERTED: %d \n' % (len(inds),))

print('Ontology LOADED\n')

# ----------------------------------------------------------

features = classes.copy()

with onto:
    for p in range(len(oprops)):
        # dom_p = oprops[p].domain
        # ran_p = oprops[p].range
        dom_p = oprops[p].domain if not (oprops[p].domain == []) else [Thing]
        ran_p = oprops[p].range if not (oprops[p].range == []) else [Thing]
        # print(dom_p,ran_p)
        new_feature = types.new_class('Some_' + oprops[p].name + '_'
                                      + 'range', (Thing,))  # dom_p[0]
        new_feature.equivalent_to = [oprops[p].some(ran_p[0])]
        features.append(new_feature)
        # new_cfeature = types.new_class('Some_' + oprops[r].name + '_Non_' + ran_r[0].name, (dom_r[0],))
        # new_cfeature.equivalent_to = [oprops[r].some(Not(ran_r[0]))]
        # features.append(new_cfeature)

with onto:
    for p in range(len(dprops)):
        dom_p = dprops[p].domain if not (dprops[p].domain == []) else [Thing]
        # check next: Thing?
        ran_p = dprops[p].range if not (dprops[p].range == []) else [Thing]
        # print(dom_p,ran_p)
        new_feature = types.new_class('Some_' + dprops[p].name + '_'
                                      + 'range', (Thing,))  # dom_p[0]
        new_feature.equivalent_to = [dprops[p].some(ran_p[0])]
        features.append(new_feature)


print('FEATURES (%d):' % (len(features)))
for i in range(len(features),):
    print(i, ": ", features[i])

print('\nExtracting Individuals per Feature')

c_features = []

with onto:
    for a_class in features:
        complement = types.new_class('Non_' + a_class.name, (Thing,))
        complement.equivalent_to = [Not(a_class)]
        c_features.append(complement)

    # sync_reasoner_pellet(infer_property_values=True,
    #                   infer_data_property_values=True,
    #                   debug=0)
    sync_reasoner_pellet(debug=0)

for f in features:
    inds = inds + list(f.instances())
    # print(f.instances())
    # print(inds)

# now sorted by name
inds = sorted(set(inds), key=lambda an_ind: an_ind.name)

# for j, ind in enumerate(inds): print('%5d: %s' % (j, ind) )
# print('\n#individuals inferred: %d \n' % (len(inds),))

v = [0, 0.5, 1]
NEG = 0
UNL = 1
POS = 2

# features x inds matrix
pi = np.full((len(features), len(inds)), v[UNL])

for f in range(len(features)):
    # print(f,'\t')
    f_instances = set(features[f].instances())
    c_instances = set(c_features[f].instances())
    for ind in f_instances:
        i = inds.index(ind)
        pi[f, i] = v[POS]
    for ind in c_instances:
        i = inds.index(ind)
        pi[f, i] = v[NEG]

X = pi.T

print("\nFEATURE SELECTION (were %d)" % (X.shape[1]))
fs = VarianceThreshold(threshold=0.1) # was 0.1
X = fs.fit_transform(X)
# print(fs.variances_)
print('X: ', X.shape)
print(fs.get_feature_names_out())


# LEARNERS -------------------------------------------------------------------------------------------------

################################################## For LR ##################################################

lr = LogisticRegression(C=0.01, penalty='l1', multi_class='multinomial', solver='saga', max_iter=200)

############################################# For MBM with bnb #############################################

from BNB import BNB
bnb = BNB()

############################################# For MBM with EM ##############################################

max_it = 200
min_change = 0.000001

from BNB_EM import BNB_EM
mbm_em = BNB_EM(max_it, min_change, is_HBM=False)

####################################### For HBM with VVBBMM + MBM_EM #######################################

# INIT NEW MBM_EM
h_mbm_em = BNB_EM(max_it, min_change, is_HBM=True)

from mixture import VBBMM

# INIT OF MIXTURE, BELOW THERE IS A NEW ONE

n_comp_init = 2 # Random number, may will change with new optimal one in grid_search
vbbmm = VBBMM(n_components=n_comp_init, n_init=10)
mixture_hbm = Pipeline(steps=[("vbbmm", vbbmm), ("clf", h_mbm_em)])

# END LEARNERS ----------------------------------------------------------------------------------------------


models = {'MBM' : bnb,
          'MBM_EM': mbm_em,
          'HBM': mixture_hbm,
          'LR': lr}

m_scoring = {'P': make_scorer(precision_score, labels = [1, 0], average = 'weighted', zero_division = 1),
             'R': make_scorer(recall_score, labels = [1, 0], average = 'weighted'),
             'F1': make_scorer(f1_score, labels = [1, 0], average = 'weighted', zero_division = 1)}


# PROBLEMS MAIN LOOP

print('\n...loading TARGET CLASSES')

filename = onto.name + "-t.nt"
target_onto = get_ontology(filename).load()


targets = np.array(sorted(set(target_onto.search(iri='*#Class_*')), key=(lambda x: x.name)))
ctargets = np.array(sorted(set(target_onto.search(iri='*#cClass_*')), key=(lambda x: x.name)))

print(targets, "\n\n")
# print(ctargets, "\n\n")

with target_onto:
    # for target_class in targets:
    #     ctarget_class = types.new_class('c' + target_class.name, (Thing,))
    #     ctarget_class.equivalent_to = [Not(target_class)]
    #
    sync_reasoner_pellet() #debug=0


averages = np.empty((len(models), len(m_scoring), len(targets)))

with onto:

    n_comps = range(1,10) # min = 2
    param_grid = {'vbbmm__n_components': n_comps}
    grid_search = GridSearchCV(mixture_hbm, param_grid) # default cv = 5

    for n in range(len(targets)):
        target_class = targets[n]
        ctarget_class = ctargets[n]

        y = np.full((pi.shape[1],), UNLABEL)

        print('\n\nTarget Class: ', target_class)
        print('≡', target_class.equivalent_to)

        pos = set(target_class.instances())
        num_pos = len(pos)
        neg = set(ctarget_class.instances())
        num_neg = len(neg)
        print("\t pos: %d\t neg: %d\t unl: %d "
              % (num_pos, num_neg, y.shape[0] - num_pos - num_neg))
        for ind in pos:
            y[inds.index(ind)] = v[POS]
        for ind in neg:
            y[inds.index(ind)] = v[NEG]

        # NEW ONE MIXTURE
        # the code below substitude the former mixture model with the new one that implement best params

        grid_search.fit(X,y)
        X = np.nan_to_num(X, nan=v[UNL]) # After grid_search unknow values may be NaN,
                                         # so i re-transform them in unlabeled values = (0.5)
        best_params = grid_search.best_params_

        new_vbbmm = VBBMM(n_components=best_params.get('vbbmm__n_components'), n_init=10)
        new_mixture_hbm = Pipeline(steps=[("vbbmm", new_vbbmm), ("clf", h_mbm_em)])


        models = {'MBM' : bnb,
                  'MBM_EM': mbm_em,
                  'HBM': new_mixture_hbm,
                  'LR': lr}


# EXPERIMENTS ------------------------------------------

        sss0 = StratifiedShuffleSplit(n_splits=N_SPLITS,
                                      test_size=TEST_PORTION,
                                      random_state=SEED)

        # to remove unlabeled exs from test sets

        sss = []
        for s0 in sss0.split(X, y):
            (trn, tst) = s0
            tst = np.array([i for i in tst if y[i] > -1])
            sss.append((trn, tst))


        scores = {}

        print('\nAvg Scores - Problem #%d' % (n + 1))

        for m, m_name in enumerate(models.keys()):
            print("\nModel:", m_name)
            scores[m] = cross_validate(models[m_name], X, y,
                                       cv=sss,
                                       scoring=m_scoring,
                                       n_jobs=-1)
            for s, score in enumerate(m_scoring.keys()):
                mean = scores[m]['test_' + score].mean()
                std_dev = scores[m]['test_' + score].std()
                print("\t %.3f ± %.3f %s" % (mean, std_dev, score))
                averages[m][s][n] = mean


print('\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> FINAL RESULTS')
print('CV Splits:', N_SPLITS, '\t test set prop.:', TEST_PORTION)

for m, m_name in enumerate(models.keys()):
    print("\nModel:", m_name)
    for s, score in enumerate(m_scoring.keys()):
        mean = averages[m][s].mean()
        std_dev = averages[m][s].std()
        print("\t %.3f ± %.3f %s" % (mean, std_dev, score))

print("\n\n")

for m in models.keys():
    print("model %s: %s" % (m, models[m]))