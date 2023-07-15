#!/usr/bin/env python
# coding: utf-8

from random import randrange, seed

import types
from owlready2 import *


onto = get_ontology("lubm.owl")

reasoning.JAVA_MEMORY = 4000

rng = 501
seed(rng)
print('Seed:', rng)

MIN_EXS = 10  # minimum neg | pos examples
print('MIN_EXS:', MIN_EXS)

N_TARGETS = 10
print('N_TARGETS:', N_TARGETS)

onto.load(reload=True)
print("Base IRI: ", onto.base_iri)

classes = list(onto.classes())
print('#Classes: %d' % (len(classes)))

for i, a_class in enumerate(classes):
    print('%5d: %s' % (i, a_class))


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

'''with onto: # was all commented
     for p in range(len(dprops)):
         dom_p = dprops[p].domain if not (dprops[p].domain == []) else [Thing]
         # check next: Thing?
         ran_p = dprops[p].range if not (dprops[p].range == []) else [Thing]
         # print(dom_p,ran_p)
         new_feature = types.new_class('Some_' + dprops[p].name + '_'
                                       + 'range', (Thing,))  # dom_p[0]
         new_feature.equivalent_to = [dprops[p].some(ran_p[0])]
         features.append(new_feature)'''

print('FEATURES (%d):' % (len(features),))
print(features)

print('\nExtracting Individuals per Feature')

c_features = []

with onto:
    for f in range(len(features)):
        a_class = features[f]
        complement = types.new_class('Non_' + a_class.name, (Thing,))
        complement.equivalent_to = [Not(a_class)]
        c_features.append(complement)
    sync_reasoner_pellet(debug=0)


for f in features:
    inds = inds + list(f.instances())

inds = sorted(set(inds), key=lambda an_ind: an_ind.name)  # now sorted by name
print('\n#individuals inferred: %d \n' % (len(inds),))


# QUERY loop

N_CANDIDATES = 20
targets = []
random_indices = []

target_onto_name = "http://www.example.org/" + onto.name + "/targets"
target_onto = get_ontology(target_onto_name)

with target_onto:
    #Target = types.new_class("Target", (Thing,))

    while len(targets) < N_TARGETS: #main loop

        Target = types.new_class("Target", (Thing,))

        print('\n>>>>> Targets (current # = %d/%d)\n' % (len(targets), N_TARGETS))

        # geneate N_CANDIDATES target classes

        candidates = []
        ccandidates = []
        print('\nGenerating next %d candidates...\n' % (N_CANDIDATES))
        for j in range(N_CANDIDATES):
            # generate j-th target

            while True:
                random_0 = randrange(0, len(features))
                random_1 = randrange(0, len(c_features))
                if not ((random_0, random_1) in random_indices):
                    break
            random_indices.append((random_0, random_1))
            # define candidate and its complement 
            # with target_onto:
            t_name = 'Class_' + str(random_0) + '_' + str(random_1)
            candidate = types.new_class(t_name, (Target,))
            candidate.equivalent_to = [features[random_0] | features[random_1]]
            ccandidate = types.new_class('c' + t_name, (Thing,))
            ccandidate.equivalent_to = [ Not(candidate) ]
            print(candidate)
            print('≡', candidate.equivalent_to)
            candidates.append(candidate)
            ccandidates.append(ccandidate)

        # find individuals belonging to candidate targets
        print('\nsyncing...\n')
        sync_reasoner_pellet([onto, target_onto], debug=1) #
        print('\n...done.\n')

        for candidate, ccandidate in zip(candidates, ccandidates):
            pos = set(candidate.instances())
            num_pos = len(pos)
            neg = set(ccandidate.instances())
            num_neg = len(neg)
            print(candidate)
            print('≡', candidate.equivalent_to)
            print("\t pos: %d\n\t neg: %d\n\t unl: %d"
                  % (num_pos, num_neg,
                     len(inds) - num_pos - num_neg))
            if len(targets) == N_TARGETS:
                candidates.remove(candidate)
                ccandidates.remove(ccandidate)
                destroy_entity(candidate)
                destroy_entity(ccandidate)
                break
            else:
                if num_pos < MIN_EXS or num_neg < MIN_EXS:
                    candidates.remove(candidate)
                    ccandidates.remove(ccandidate)
                    destroy_entity(candidate)
                    destroy_entity(ccandidate)
                    print('---------------------------------- REMOVED')
                else:
                    targets.append(candidate)
                    print('---------------------------------- SELECTED')
            print()
        candidates = targets
        print('\nValid candidates found (%d):\n%s \n' % (len(candidates), candidates))

        #targets = targets + candidates

# truncate to first N_TARGETS

# -------------------------------------------------------------------------
# SAVING ON FILE
print('\n\nTarget Classes:\n', targets)
filename = onto.name + "-t.nt"
target_onto.save(file=filename)
print('\nwritten to:', filename)
target_onto.destroy()
# onto.destroy()