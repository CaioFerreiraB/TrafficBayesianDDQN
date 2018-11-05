
from flow.core.generator import Generator

class OneJunctionGenerator(Generator):

	def specify_routes(self, net_params):
		"""Specify the routes used in the scenario"""

		rts = {
				'bottom' : 'bottom up',
				'right' : 'right left',

			  }

		return rts

	def generate_net(self, net_params, traffic_lights):
		"""See parent class.

		The network file is generated from the .osm file specified in
		net_params.osm_path
		"""
		# name of the .net.xml file (located in cfg_path)
		self.netfn = net_params.netfile

		# collect data from the generated network configuration file
		edges_dict, conn_dict = self._import_edges_from_net()

		return edges_dict, conn_dict