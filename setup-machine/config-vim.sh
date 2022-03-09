#!/bin/bash

git clone https://github.com/altercation/solarized.git
mkdir -p ~/.vim/colors
cp solarized/vim-colors-solarized/colors/solarized.vim ~/.vim/colors/
git clone https://github.com/fugalh/desert.vim.git
cp desert.vim/colors/desert.vim ~/.vim/colors/
cp vimrc ~/.vimrc

git clone https://github.com/seebi/dircolors-solarized.git
cp -r dircolors-solarized ~/.dir_colors

cp tmux.conf ~/.tmux.conf

git clone https://github.com/opensourcedesign/fonts.git
mkdir -p ~/.fonts/freefont
cp fonts/gnu-freefont_freemono/FreeMono.ttf  ~/.fonts/freefont/
