from recap_am.model.aif import Node, NodeCategory, Graph, Edge

def parser(input):

    """
    Transformation of exported discussions from kialo.com to the AIF format.
    To export manually (this step can be automated which is prohibited by the ToS.):
        - open a discussion on kialo.com
        - click the discussion menu button on the upper left corner (three horizontal bars)
        - click the discussion settings button
        - scroll down and click Export Discussion

    Example input:
        "Discussion Title: Programing is still a skill not many people are good at. How do we change this?

        1. Programing should be mandatory in school to reduce.. (--> MAJOR CLAIM)
        1.1. Con: Kids may not be interested in a carreer in tech.
        1.1.2. Con: In the future every field will include tech to a certain degree.
        1.2. Pro: Understanding programing opens up opportunities for interesting projects and clubs.
        1.1.2. Pro: Children can use their time creating applications or working with tiny roboters."
    """
    # Read lines of the text File.
    lines = []
    with open(input, "r", encoding="utf8") as file:
        for line in file:
            lines.append(line)

    # Remove first two lines of document since Discussion Title and the second lines do not contain necessary information.
    lines = lines[2:]
    level, mc_text = lines[0].split(" ", maxsplit=1)
    # Remove major claim
    lines = lines[1:]

    # delete lines that do not have levels on them, future TODO: add lines to last element with level in chat
    delete_positions = []
    for i, line in enumerate(lines):
        if line[:2] != "1.":
            delete_positions.append(i)
            # print("+")
    for i in delete_positions:
        lines.pop(i)

    # Processing info
    pairs = {}
    pairs["level"] = []
    pairs["stance"] = []
    pairs["text"] = []
    pairs["id"] = []

    # Processing info
    pairs = {}

    for i, line in enumerate(lines):
        level, stance, text = line.split(" ", maxsplit=2)
        text = text.strip("\n")
        pairs[level] = [i + 1, stance[:-1], text]

    # used later to create the scheme nodes within

    # some nodes reference other nodes instead of writing the same text down
    # we are querying all pairs and turn the reference into text again
    # Example:
    # '1.10.18.': [521, 'Con', '-> See 1.10.5.3.']}
    # '1.10.5.3.': [395, 'Pro', "People who believe in God often ...."]
    for level, node_info in pairs.items():
        if "-> See" in node_info[2]:
            trash, reference = node_info[2].split(sep="See ")
            pairs[level][2] = pairs[reference][2]

    edges = [Edge]
    nodes = [Node]
    # add MC to set of nodes
    major_claim_node = Node(
        key=0, _text=mc_text, category=NodeCategory("I"), major_claim=True
    )
    nodes.append(major_claim_node)

    for i, (level, node_info) in enumerate(pairs.items()):

        parent_level, trash = level[:-1].rsplit(sep=".", maxsplit=1)
        parent_level += "."

        ## CREATE A SCHEME NODE
        current_scheme_id = len(lines) + 1 + i
        scheme_node = Node(
            key=current_scheme_id,
            _text=node_info[2],
            category=NodeCategory("RA" if node_info[1] == "Pro." else "CA"),
        )
        nodes.append(scheme_node)

        # CREATE THE CHILD I NODE
        child_I_node = Node(
            key=node_info[0], _text=node_info[2], category=NodeCategory("I")
        )
        nodes.append(child_I_node)

        ### (Parent) I Node <-- (EDGE  <-- SCHEME NODE --> EDGE) --> (Child) I Node
        if parent_level == "1.":
            child_2_scheme_edge = Edge(start=node_info[0], end=current_scheme_id)
            scheme_2_parent_edge = Edge(start=current_scheme_id, end=0)
        else:
            child_2_scheme_edge = Edge(start=node_info[0], end=current_scheme_id)
            scheme_2_parent_edge = Edge(
                start=current_scheme_id, end=pairs[parent_level][0]
            )

        edges.append(child_2_scheme_edge)
        edges.append(scheme_2_parent_edge)

    test_Graph = Graph(
        key="test", nodes=nodes, edges=edges, participants=None, analysis=None
    )
    print(test_Graph.to_aif())


if __name__ == "__main__":
    test = []
    parser("/app/recap_am/preprocessing/test_files/god_example.txt")
