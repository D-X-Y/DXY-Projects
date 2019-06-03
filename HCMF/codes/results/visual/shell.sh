probs="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

for prob in ${probs}
do
	pdfcrop curve-${prob}.pdf curve-${prob}.pdf
	pdfcrop eror-${prob}.pdf  eror-${prob}.pdf
	pdfcrop grot-${prob}.pdf  grot-${prob}.pdf
	pdfcrop rcec-${prob}.pdf  rcec-${prob}.pdf
	pdfcrop ours-${prob}.pdf  ours-${prob}.pdf
done
