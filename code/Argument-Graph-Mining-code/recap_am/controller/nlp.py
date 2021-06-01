import logging
from typing import Any, Dict, List
import nltk
import spacy
from spacy.tokens import Doc
import datetime

from recap_am.model.config import Config

logger = logging.getLogger(__name__)
config = Config.get_instance()

# Use this attribute as nlp .parse("text")
parse = spacy.load(config["nlp"]["spacy_model"])
Doc.set_extension("key", default=datetime.datetime.now().isoformat("_"))
