#!/bin/bash

git clone https://github.com/altercation/solarized.git
mkdir -p ~/.vim/colors
cp solarized/vim-colors-solarized/colors/solarized.vim ~/.vim/colors/
rm -rf solarized

git clone https://github.com/fugalh/desert.vim.git
cp desert.vim/colors/desert.vim ~/.vim/colors/
rm -rf desert.vim
cp meta/vimrc ~/.vimrc

# git clone https://github.com/seebi/dircolors-solarized.git
# cp -r dircolors-solarized ~/.dir_colors
# rm -rf dircolors-solarized

cp meta/tmux.conf ~/.tmux.conf

git clone https://github.com/opensourcedesign/fonts.git
mkdir -p ~/.fonts/freefont
cp fonts/gnu-freefont_freemono/FreeMono.ttf  ~/.fonts/freefont/
rm -rf fonts

# Git Completion System
curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
