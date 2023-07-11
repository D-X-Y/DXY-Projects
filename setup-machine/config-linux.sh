#!/bin/bash
script_dir=$(realpath $(dirname "$0"))
echo "The script directory is ${script_dir}"

# VIM
git clone https://github.com/altercation/solarized.git
mkdir -p ~/.vim/colors
cp ${script_dir}/solarized/vim-colors-solarized/colors/solarized.vim ~/.vim/colors/
rm -rf ${script_dir}/solarized
git clone https://github.com/fugalh/desert.vim.git
cp desert.vim/colors/desert.vim ~/.vim/colors/
rm -rf ${script_dir}/desert.vim
cp ${script_dir}/meta/vimrc ~/.vimrc

# git clone https://github.com/seebi/dircolors-solarized.git
# cp -r dircolors-solarized ~/.dir_colors
# rm -rf dircolors-solarized

# TMUX
cp meta/tmux.conf ~/.tmux.conf

git clone https://github.com/opensourcedesign/fonts.git
mkdir -p ~/.fonts/freefont
cp fonts/gnu-freefont_freemono/FreeMono.ttf  ~/.fonts/freefont/
rm -rf fonts

# Git Completion System
curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash

# Config GitHub
git config --global user.email "dxy@augment.code"
git config --global user.name "D-X-Y"
git config --global push.default matching
git config --global push.default simple

# kubectl : https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
