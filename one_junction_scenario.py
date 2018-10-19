from flow.scenarios.netfile.scenario import NetFileScenario

class OneJunctionScenario (NetFileScenario):

	def specify_edge_starts(self):

		edge_start = [('L3', 0), ('L5', 80)]

		return edge_start

	def specify_intersection_edge_starts(self):
		"""See parent class."""
		intersection_edgestarts = [("E0", 0)]

		return intersection_edgestarts