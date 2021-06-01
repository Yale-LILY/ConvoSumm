

from sentence_transformers import SentenceTransformer
import scipy.cluster.hierarchy as hcluster

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.'
          ]
corpus_embeddings = embedder.encode(corpus)

# Perform hierarchical clustering
cosine_threshold = 0.7
cluster_assignment = hcluster.fclusterdata(
    corpus_embeddings, 
    1-cosine_threshold, 
    criterion="distance", 
    metric="cosine", 
    method="average"
    )

num_clusters = len(set(cluster_assignment))
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id-1].append(corpus[sentence_id])


for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
