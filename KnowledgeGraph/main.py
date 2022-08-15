"""
ENTIRE CODE IS USELESS
"""

from rdflib import Graph, term, Literal
from rdflib.namespace import XSD
import pandas as pd

RDF_URL = "http://wordnet-rdf.princeton.edu/wn30/"
labels_path = "../Inputs/Labels/wordnet_details.txt"             #path to labels
super_class_label_path = "../Inputs/Labels/Superclasses.txt"

hypernym = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#hypernym')
meronym = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#meronym')
sense = term.URIRef('http://www.w3.org/ns/lemon/ontolex#sense')

#Predicates that will be eliminated from all Triples of every word
hyponym = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#hyponym')
antonym = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#antonym')
sameAs = term.URIRef('http://www.w3.org/2002/07/owl#sameAs')
subject = term.URIRef('http://purl.org/dc/terms/subject')
pos = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#partOfSpeech')
derivation = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#derivation')
has_domain_topic = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#has_domain_topic')
holo_member = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#holo_member')
domain_topic = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#domain_topic')
mero_part = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#mero_part')
mero_substance = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#mero_substance')

value = term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#value')
definition = term.URIRef('http://wordnet-rdf.princeton.edu/ontology#definition')
label_predicate_uri = term.URIRef('http://FAKE-rdf.label./ontology#label') #will use it for the purpose of node class
reverse_label_predicate_uri = term.URIRef('http://FAKE-rdf.revlabel./ontology#label')
label_object_uri = 'http://wordnet-rdf.princeton.edu/rdf/label_name/'

wordnet_synset_id_uri = 'http://wordnet-rdf.princeton.edu/rdf/id/'
wordnet_id_uri = 'http://wordnet-rdf.princeton.edu/rdf/pwn30/'
superclass_uri = term.URIRef('http://FAKE-rdf.superclass./ontology#superclass')


context_Graph_ = Graph()
train_Graph_ = Graph()

label_data_frame = pd.read_csv(labels_path, delimiter=" ")
super_class_labels_data = pd.read_csv(super_class_label_path, delimiter=" ")
super_class_synset_id = dict()


def make_label_graph(subject, label_name):
    triple = (term.URIRef(subject), label_predicate_uri, term.URIRef(label_object_uri + label_name))
    train_Graph_.add(triple)


def get_hypernyms(h_list,super_class):
    hypernym_Graph = Graph()

    for id in h_list:
        gg, _ = clean_graph(id, False)

        for s, p, o in gg:
            if p == hypernym:
                gg.remove((s, p, o))
                o = super_class_synset_id[super_class.lower()]
                gg.add((s,p,o))

        for tripl in gg:
            hypernym_Graph.add(tripl)

    return hypernym_Graph


def clean_graph(id, flag):
    temp_ = Graph()
    temp_.parse(id, format='application/rdf+xml')
    hypernym_list = []

    for s,p,o in temp_:
        if p == hyponym or p == antonym or p == sameAs or p == subject \
                or p == pos or p == derivation or p == has_domain_topic \
                or p == holo_member or p == domain_topic:
            temp_.remove((s, p, o))

        elif p == mero_part or p == mero_substance:
            temp_.remove((s,p,o))
            p = meronym
            temp_.add((s,p,o))

        elif type(o) == Literal and p != subject:
            temp_.remove((s, p, o))
            literal_val = o.value
            if type(literal_val) == str:
                dtype = XSD.string
            elif type(literal_val) == int:
                dtype = XSD.decimal

            o = Literal(lexical_or_value=literal_val, datatype=dtype)
            temp_.add((s, p, o))

        elif p == hypernym and flag:
            hypernym_list.append(o)

    return temp_, hypernym_list


def get_synset_id(grph,label__,wrdnet_id):
    wd_ = term.URIRef(wordnet_id_uri+wrdnet_id+"#"+label__ + "-n")
    wd_temp = wordnet_id_uri+wrdnet_id+"#"+label__+"-"
    for s,p,o in grph:
        if s == wd_ and p == sense:
            dd = o.split(wd_temp)[1]
            break
    super_class_synset_id[label__] = term.URIRef(wordnet_synset_id_uri + dd)


def make_superclasses_graph(labels_):
    temp_ = Graph()

    for row in labels_.iterrows():
        wd_id = row[1][0]
        label_name = row[1][1]
        id = RDF_URL + wd_id + ".rdf"
        if label_name == "object":
            rdf_id_of_object = wordnet_id_uri + wd_id
        temp_.parse(id, format='application/rdf+xml')
        get_synset_id(temp_,label_name,wd_id)

    for s, p, o in temp_:
        if p == hypernym:
            if s != term.URIRef(rdf_id_of_object):
                temp_.remove((s, p, o))
                o = term.URIRef(rdf_id_of_object)
                temp_.add((s, p, o))
            else:
                temp_.remove((s, p, o))

        elif type(o) == Literal and p != subject:
            temp_.remove((s, p, o))
            literal_val = o.value
            if type(literal_val) == str:
                dtype = XSD.string
            elif type(literal_val) == int:
                dtype = XSD.decimal

            o = Literal(lexical_or_value=literal_val, datatype=dtype)
            temp_.add((s, p, o))

        elif p == hyponym or p == antonym or p == sameAs or p == subject\
                or p == pos or p == derivation or p == has_domain_topic \
                or p == holo_member or p == domain_topic:
            temp_.remove((s, p, o))

        elif p == mero_part or p == mero_substance:
            temp_.remove((s,p,o))
            p = meronym
            temp_.add((s,p,o))

    return  temp_


def construct_mini_wordnet_knowledge_graph(labels_):
    for row in labels_.iterrows():
        id = row[1][0].split("n")[1]
        label_name = row[1][1]
        supclass = row[1][2]
        id = RDF_URL + id + "-n.rdf"
        gg, h_list = clean_graph(id,True)

        hyper_gg = get_hypernyms(h_list, supclass)

        for tripl in gg:
            context_Graph_.add(tripl)

        for tripl in hyper_gg:
            context_Graph_.add(tripl)


if __name__ == "__main__":
    sup_graph = make_superclasses_graph(super_class_labels_data)

    for tripl in sup_graph:
        context_Graph_.add(tripl)

    construct_mini_wordnet_knowledge_graph(label_data_frame)
    context_Graph_.parse("../Inputs/context_MWKG.nt")

    context_Graph_.serialize("../Inputs/context_MWKG.nt", format="ttl", encoding='utf-8')
    context_Graph_.close()

    # train_Graph_.serialize("../Inputs/train_MWKG.nt", format="nt", encoding='utf-8')
    # train_Graph_.close()





