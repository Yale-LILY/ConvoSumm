
from collections import defaultdict


#with open("stackexchange_newnocomment/stackexchange.val.source.remove_markers_simple_separator.nocomment.extrasep") as inputf:
#    for count, line in enumerate(inputf):
#        print(count)
#        line = line.replace("</s></s></s></s></s></s>", "</s></s>")
#        line = line.replace("</s></s></s></s>", "</s></s>").replace("</s></s></s>", "</s></s>")
#        line = line.strip().replace("</s></s></s>", "</s></s>").replace("</s></s></s>", "</s></s>")
#        if line[-4:] == "</s>":
#            line = line[:-4]
#        if line[-4:] == "</s>":
#            line = line[:-4]
#        line_split = line.split("</s></s>")
#        info = line_split[0]
#        rest_of_lines = [x for x in line_split[1:] if len(x.strip()) > 0]
#        queries = [x.split("</s>")[1].strip() for x in rest_of_lines]
#        if len(queries) == 0:
#            print("HI")
#        queries = []
#        subject_info = []
#        for email in rest_of_lines:
#            try:
#                cur_subject_info, query = email.split("</s>")
#            except:
#                import pdb;pdb.set_trace()
#            query = query.strip()
#            if len(query) == 0:
#                continue
#            queries.append(query.strip())
#            subject_info.append(cur_subject_info.strip())

with open("/private/home/alexfabbri/convosumm/Argument-Graph-Mining/corpus/val.source") as inputf:
    for count, line in enumerate(inputf):
        print(count)
        line = line.replace("</s></s></s></s></s></s>", "</s></s>")
        line = line.replace("</s></s></s></s>", "</s></s>").replace("</s></s></s>", "</s></s>")
        line = line.strip().replace("</s></s></s>", "</s></s>").replace("</s></s></s>", "</s></s>")
        if line[-4:] == "</s>":
            line = line[:-4]
        if line[-4:] == "</s>":
            line = line[:-4]
        line_split = line.split("</s>")

        queries = []
        subject_info = []
        person2utt = defaultdict(list)
        for email in line_split:
            try:
                email_split = email.split(":")
                cur_subject_info = email_split[0].strip()
                query = ":".join(email_split[1:])
                query = query.strip()
                if len(query) == 0:
                    continue
                person2utt[cur_subject_info].append(query)
            except:
                import pdb;pdb.set_trace()
        assert len(queries) == len(subject_info)

        for cur_subject_info, query in person2utt.items():
            subject_info.append(cur_subject_info)
            queries.append(" ".join(query))
