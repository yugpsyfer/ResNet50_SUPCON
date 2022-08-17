"""
THIS CODE WAS JUST A HELPER IN EXPERIMENTATION AND CAN BE IGNORED
"""
#
# from rdflib import Graph
# import rdflib
#
# graph = Graph()
# g1 = Graph()
# uri = 'http://www.IIITD_WN18RR/'
#
# #
# # pred_set = set()
# #
# # with open("../WN18RR.nt", 'r') as fp:
# #     for j in fp.readlines():
# #         j_l = j.split("\t")
# #         subj = rdflib.URIRef(uri+j_l[0])
# #         pred = rdflib.URIRef(uri+"owl#"+j_l[1])
# #         obj = rdflib.URIRef(uri+j_l[2].split("\n")[0])
# #         graph.add((subj, pred, obj))
# #
# #
# #
# # graph.serialize(destination="./WN18RR.nt",format='nt', encoding='utf-8')
#
# dict = {'ladybug': '02165456', 'snorkel': '01963795', 'cannon': '02950826', 'organ': '08349350', 'trifle': '00711932', 'lion': '09752795', 'ear': '05320899', 'dome': '03220513', 'parallel_bars': '03888605', 'file': '01920048', 'reel': '04067472', 'orange': '07747607', 'missile': '03773504', 'goose': '01457079', 'crate': '03127925', 'yawl': '01046932', 'hair_slide': '03476684', 'clog': '01709931', 'unicycle': '01935846', 'slot': '04243727', 'cuirass': '03146219', 'school_bus': '04146614', 'bolete': '13054560', 'ipod': '03584254', 'barrel': '01502540', 'dalmatian': '02110341', 'spider_web': '04275363', 'chime': '02182342', 'king_crab': '07788435', 'three-toed_sloth': '02457408', 'lipstick': '03676483', 'jellyfish': '01910747', 'tank': '04389033', 'stage': '04296562', 'oboe': '03838899', 'coral_reef': '09256479'}
# # keys = set()
# #
# # for i in dict.values():
# #     keys.add(rdflib.URIRef(uri+i))
#
# # knows_query = """SELECT * WHERE {"""+ +"""?p ?b . ?b ?p1 ?o .}"""
#
#
# graph.parse(r'C:\Users\yugss\Documents\Github_Repos\ResNet50_SUPCON\KnowledgeGraph\utilities\WN18RR.nt', format='nt')
#
#
#
# for s,p,o in graph:
#     if s == rdflib.URIRef(uri+"02165456") or o == rdflib.URIRef(uri+"02165456"):
#         g1.add((s,p,o))
#
# g1.serialize(destination="./WN18RR-subset.nt",format='nt', encoding='utf-8')
#
# # for kys in keys:
# #     k = graph.query("SELECT * WHERE { <"+str(kys)+"> ?p ?b . "
# #                                                   "?b ?p1 ?o .}")
# #
# #     for i in k:
# #         print(i)