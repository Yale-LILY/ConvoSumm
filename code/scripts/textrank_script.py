#from summa.summarizer import summarize
import sys

domain = sys.argv[1]

if domain == "nyt":
    output_len = 79
elif domain == "reddit":
    output_len = 65
elif domain == "stackexchange":
    output_len = 73
elif domain == "email":
    output_len = 75
else:
    exit()


from gensim.summarization.summarizer import summarize 
filename = f"{domain}.test.source.remove_markers_simple_separator"
with open(filename) as inputf, open(filename + ".textrank", "w") as outputf:
    for count, line in enumerate(inputf):
        if count % 50 == 0:
            print(count)
        line = line.strip()
        output_summary = summarize(line, word_count=output_len).replace("\n", " ") 
        outputf.write(output_summary + "\n")


        
