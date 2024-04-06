fastchat_dir="/root/FastChat"

if [ ! -d $fastchat_dir ]
then
  git clone https://github.com/lm-sys/FastChat.git
  if [ ! -d $fastchat_dir ]
    then
      echo "Konnte Fastchat nicht herunterladen."
      exit 1
    else
      echo "Fastchat-Repo heruntergeladen"
      cd FastChat
      pip3 install --upgrade pip  # enable PEP 660 support
      pip3 install cmake lit
      pip3 install -e ".[model_worker,webui]"
  fi
fi

tmux split-window -h python3 -m fastchat.serve.controller --host 0.0.0.0

tmux split-window -v -p 66 python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002,Llama-2-13b-german" --model-path jphme/em_german_7b_v01 --host 0.0.0.0

tmux split-window -v python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 7860


# worker 0
#CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
#CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001