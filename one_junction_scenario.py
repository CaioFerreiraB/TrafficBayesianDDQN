from flow.scenarios.netfile import NetFileScenario

class OneJunctionScenario (NetFileScenario):
	def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
		"""Generate a user defined set of starting positions.
		For this simulation, we just want the starting positions of the vehicles to be on the begin of the right and bottom edges

        Parameters
        ----------
        initial_config : InitialConfig type
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network
        kwargs : dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions : list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
            list of start lanes
        """
		startpositions = [('bottom', 30), ('bottom', 20), ('bottom', 10), ('bottom', 0), ('right', 10)]
		startlanes = [0, 0, 0, 0, 0]

		return startpositions, startlanes

