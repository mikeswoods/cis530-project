import subprocess
import sys
from logging import info
from project.utils import files

################################################################################
# Command running and output capture utility functions
################################################################################

_CORENLP_CMD = 'java -cp "*" -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP'


def run_command(*args, **kwargs):
	"""
	"""
	if 'stderr' in kwargs:
		if kwargs['stderr'] == 'pipe':
			stderr = subprocess.PIPE
		else:
			stderr = kwargs['stderr']
	else:
		stderr = sys.stderr

	pipe       = subprocess.Popen(*args, stdout=subprocess.PIPE, stderr=stderr)
	(out, err) = pipe.communicate()
	return (out, err)


def run_corenlp(working_dir
	           ,input_files
	           ,output_dir=None
	           ,annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner']):
	"""
	"""
	args = ['java', '-cp', '*', '-Xmx1g', 'edu.stanford.nlp.pipeline.StanfordCoreNLP']

	if (output_dir is not None):
		args += ['-outputDirectory', output_dir]

	args += ['-annotators', ','.join(annotators)]

	if isinstance(input_files, str):
		input_files = [input_files]

	with files.cd(working_dir):

		with files.write_temp(files.to_lines(input_files)) as file_list:

			args += ['-filelist', file_list]

			info('$ '+' '.join(args))

			(out, err) = run_command(*[args])
