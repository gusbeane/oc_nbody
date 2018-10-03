import agama

class agama_wrapper(object):
	def __init__(self, index):
		
	def __init__(self, dm_file, bar_file):
		self.pdark = agama.Potential(dm_file)
		self.pbar = agama.Potential(bar_file)
		self.potential = agama.Potential(self.pdark, self.pbar)

	def actions(self, poslist, vlist):
