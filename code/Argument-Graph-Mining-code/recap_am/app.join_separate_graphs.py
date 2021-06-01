from __future__ import absolute_import, annotations
import sys

import io
import logging
import multiprocessing
import os
import traceback
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from zipfile import ZipFile

import flask
import pendulum
import recap_argument_graph as ag
import werkzeug
from sklearn.model_selection import ParameterGrid

from recap_am.adu import run_task
from recap_am.controller.preprocess import prep_production
from recap_am.evaluation import evaluator
from recap_am.model.config import Config
from recap_am.model.query import Query
from recap_am.model.statistic import Statistic, Statistics
from recap_am.relation import construct_graph
from recap_am.relation.controller import attack_support
import networkx as nx

logging.basicConfig(level=logging.WARNING)

log = logging.getLogger(__package__)
log.setLevel(logging.INFO)

config = Config.get_instance()

import warnings

warnings.filterwarnings("ignore")


def run_server() -> None:
    """Start a flask server."""

    app = flask.Flask(__package__, root_path=str(Path(__file__).resolve().parent))
    app.config.update(TEMPLATES_AUTO_RELOAD=True, FLASK_ENV="development")
    app.secret_key = os.urandom(16)

    @app.route("/", methods=["POST", "GET"])
    def index():
        stats = None

        if flask.request.method == "POST":
            try:
                query_files = flask.request.files.getlist("query-files")
                _update_config(flask.request.form, from_flask=True)
                stats = run(query_files)
            except Exception:
                flask.flash(traceback.format_exc(), "error")

        return flask.render_template("index.html", config=config, statistics=stats)

    @app.route("/download/<folder>")
    def download(folder):
        out_path = Path(config["path"]["output"], folder)

        if out_path.exists():
            data = io.BytesIO()

            with ZipFile(data, mode="w") as z:
                for file in out_path.iterdir():
                    z.write(file, file.name)

            data.seek(0)
            # shutil.rmtree(out_path)

            return flask.send_file(
                data,
                mimetype="application/zip",
                as_attachment=True,
                attachment_filename=f"{folder}.zip",
            )

        flask.flash("The requested file is not available.", "error")
        return flask.render_template("index.html", config=config, statistics=None)

    log.info("If run via docker, the address is http://localhost:8888.")
    app.run(host=config["flask"]["host"], port=config["flask"]["port"])


def evaluate() -> None:
    param_grid = {
        "mc-method": ["centroid", "first", "pairwise", "relations"],
        "relation-threshold": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "relation-method": ["adu_position", "flat_tree", "pairwise_comparison"],
    }
    grid = ParameterGrid(param_grid)
    timestamp = _timestamp()

    params = [(grid_entry, timestamp) for grid_entry in grid]

    if config["debug"]:
        for param in params:
            _single_eval(*param)

    else:
        with multiprocessing.Pool() as pool:
            pool.starmap(_single_eval, params)


def _single_eval(params: t.Mapping[str, t.Any], timestamp: str) -> None:
    log.info(f"Evaluating with params: {params}")
    _update_config(params)

    run(
        timestamp=timestamp,
        subfolder="-".join((str(value) for value in params.values())),
    )


def _timestamp() -> str:
    return pendulum.now().format("YYYY-MM-DD-HH-mm-ss")


def run(
    query_files: t.Optional[t.List[werkzeug.FileStorage]] = None,
    timestamp: t.Optional[str] = None,
    subfolder: t.Optional[str] = None,
) -> Statistics:
    """Run the argument mining process."""

    if not timestamp:
        timestamp = _timestamp()

    out_path = Path(config["path"]["output"], timestamp)

    if subfolder:
        out_path = out_path / subfolder

    out_path.mkdir(parents=True)
    stats = Statistics(timestamp)

    log.info("Loading documents.")
    start_time = timer()

    queries = Query.from_flask(query_files) or Query.from_folder(
        Path(config["path"]["input"])
    )

    for i, query in enumerate(queries):
        log.info(f"Processing '{query.name}' ({i + 1}/{len(queries)}).")

        stat = stats.new(query)

        try:
            _process_prev(query, stat, out_path)
        except Exception:
            print(traceback.format_exc())

    stats.duration = timer() - start_time
    stats.save(out_path)

    log.info("Done.")

    return stats


def _process(filename, statistic: Statistic, out_path: Path) -> None:
    """Do the processing for one input file."""

    with open(filename) as inputf, open(filename + ".graph", "w") as outputf:
        for count, line in enumerate(inputf):
            start_time = timer()
            print(count)
            line_split = line.split("</s>")
            info = line_split[0] # TODO: assumes that there is information such as forum name, question tags, which should be included in the source text but not as part of the graph

            queries = []
            subject_info = []
            for email in line_split[1:]:
                try:
                    #email_split = email.split(":")
                    #cur_subject_info = email_split[0]
                    #query = ":".join(email_split[1:])
                    # TODO: assumes that comments are separated by </s> and that each comment contains information about the speak which comes before a <s>
                    cur_subject_info, query = email.split("<s>")
                except:
                    import pdb;pdb.set_trace()
                query = query.strip()
                if len(query) == 0:
                    continue
                queries.append(query.strip())
                subject_info.append(cur_subject_info.strip())
            # Preprocessing
            docs = []
            claim_sents = []
            claim2node = {}
            graphs = []
            for query_count, query_doc in enumerate(queries):
                doc = prep_production(filename, query_doc)
                # ADU classification
                doc = run_task.run_production(doc, True)
                docs.append(doc)

                # Attack/support classification
                rel_types = []

                # Create graph with relationships
                cur_claim2node, cur_claim_sents, cur_graph = construct_graph.main(doc, rel_types, index=query_count)
                for claim, node in cur_claim2node.items():
                    claim2node[claim] = node

                claim_sents.extend(cur_claim_sents)
                graphs.append(cur_graph)
            
            claim_graph, conversation_node = construct_graph.main(doc, [], combine=True, claim2node=claim2node, claim_sents=claim_sents)
            graphs.append(claim_graph)
            graph_end2end = ag.Graph(name=doc._.key.split("/")[-1])
            for graph in graphs:
                for edge in graph.edges:
                    graph_end2end.add_edge(edge)

            gnx = graph_end2end.to_nx()
            gdict = graph_end2end.to_dict()
            gedges = list(nx.dfs_edges(gnx, conversation_node.key))
            output_text = " "
            id2text = {node['id']: node['text'] for node in gdict['nodes']}
            done = []
            for edge in gedges:
                n1, n2 = edge
                n1_text = id2text[n1]
                if n1 in done:
                    n1_text = ""
                else:
                    done.append(n1)
                n2_text = id2text[n2]
                if n1_text == "Conversation" and n2_text == "Issue":
                    continue
                elif n1_text == "Conversation":
                    output_text += "<c> " + n2_text + " "
                elif n1_text == "Issue": 
                    output_text += "<i> " + n2_text + " "
                elif n1_text == "Default Inference":
                    output_text += "<e> " + n2_text + " "
                elif n2_text == "Default Inference":
                    output_text += "<c> " + n1_text + " "
                else:
                    output_text += "<c> " + n1_text + " <e> " + n2_text + " "
            output_text = output_text.replace("\n", " ")
            outputf.write(output_text + "\n")

            statistic.duration = timer() - start_time

            suffix = filename.split("/")[-1] + str(count)
            _export(graph_end2end, out_path, suffix)

            statistic.save(out_path)


def _process_prev(query: Query, statistic: Statistic, out_path: Path) -> None:
    """Do the processing for one input file."""

    start_time = timer()

    # Preprocessing
    doc = prep_production(query.name, query.text)

    # ADU classification
    doc = run_task.run_production(doc)

    # Attack/support classification
    #rel_types = attack_support.classify(doc._.ADU_Sents)
    rel_types = []

    # Create graph with relationships
    graph_end2end = construct_graph.main(doc, rel_types)

    statistic.duration = timer() - start_time

    _export(graph_end2end, out_path, "end2end")

    if query.benchmark:
        graph_preset = evaluator.run(
            statistic, doc, rel_types, graph_end2end, query.benchmark
        )
        _export(graph_preset, out_path, "preset")

    statistic.save(out_path)


def _export(graph: ag.Graph, folder: Path, suffix: str = "") -> None:
    """Export a graph according to settings in `config`."""

    if config["export"]["json"]:
        import os
        graph.save(Path(os.path.join(folder, f"{graph.name}-{suffix}.json")))

    try:
        if config["export"]["picture"]:
            import os
            graph.render(Path(os.path.join(folder, f"{graph.name}-{suffix}.pdf")))
    except:
        pass


def _update_config(data: t.Mapping[str, t.Any], from_flask: bool = False) -> None:
    """Contents of `config.toml` is updated according to web request.

    This only works with options that can be changed without reloading the whole program.
    For example, it is not possible to change the language model as it is only loaded once during initialization.
    """

    if from_flask:
        config["export"]["json"] = bool(data.get("export-json"))
        config["export"]["picture"] = bool(data.get("export-picture"))

        config["path"]["input"] = data["input-path"]

        config["relation"]["fallback"] = data["relation-fallback"]

    config["adu"]["MC"]["method"] = data["mc-method"]
    config["relation"]["method"] = data["relation-method"]
    config["relation"]["threshold"] = float(data["relation-threshold"])

if __name__ == "__main__":
    timestamp = _timestamp()
    stat = Statistics(timestamp)

    # for full runs
    inputfname = sys.argv[1]
    out_path = sys.argv[2]
    _process(inputfname, stat, out_path)
