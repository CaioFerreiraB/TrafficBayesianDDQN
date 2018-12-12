from flow.scenarios.netfile.gen import NetFileGenerator

class OneJunctionGenerator(NetFileGenerator):

	def specify_routes(self, net_params):
		"""Specify the routes used in the scenario"""

		rts = {
				'bottom' : 'bottom up',
				'right' : 'right left',

			  }

		return rts