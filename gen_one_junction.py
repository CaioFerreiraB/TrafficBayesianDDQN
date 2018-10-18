from flow.scenarios.netfile.gen import NetFileGenerator

class OneJunctionGenerator(NetFileGenerator):

	def specify_routes(self, net_params):
		"""Specify the routes used in the scenario"""

		rts = {
				'L3' : 'L3 L6',
				'L5' : 'L5 L0'
			  }

		return rts