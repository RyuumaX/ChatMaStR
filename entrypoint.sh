#!/bin/sh

# check if knowledgebase directory exists?
DIR='./KnowledgeBase'
VECTORSTORE='chroma.sqlite3'

if [ ! -s "$DIR/$VECTORSTORE" ]; then
  # if not existing, run index_knowledgebase.py application to create knowledgebase
  echo 'entrypoint.sh::: Knowledgebase does not exist... try to create with index_knowledgebase.py'
  python3 index_knowledgebase.py -d $DIR -o $DIR
fi

# run main.py with streamlit
echo 'entrypoint.sh::: Run main.py with streamlit on port 8501'
streamlit run main.py --server.port=8501