from recap_am.relation.model.relation import RelationClass


def run_spacy(adus, relations):
    """Compute MajorClaim through classifications.

    Relations is indexed by spacy objects."""
    count = [0] * len(adus)

    for idx, adu1 in enumerate(adus):
        for adu2 in adus:
            if adu1 != adu2:
                relation = relations[adu1][adu2]

                if relation.classification != RelationClass.NONE:
                    count[idx] += relation.probability

    index_max = max(range(len(count)), key=count.__getitem__)

    return adus[index_max]


def run_str(adus, relations):
    """Compute MajorClaim through classifications.

    Relations is indexed by string objects."""
    count = [0] * len(adus)

    for idx, adu1 in enumerate(adus):
        relation_mapping = {rel.adu: rel for rel in relations[adu1.text]}

        for adu2 in adus:
            if adu1 != adu2:
                relation = relation_mapping[adu2.text]

                if relation.classification != RelationClass.NONE:
                    count[idx] += relation.probability

    index_max = max(range(len(count)), key=count.__getitem__)

    return adus[index_max]
