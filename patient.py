class Patient:

    def __init__(self, id, attr, value):
        self.id = str(id)
        self.attr = str(attr)
        self.val = str(value)


class TemporalPatient:
    def __init__(self, time_index, xvar):
        self.time_index = time_index
        self.xvar = xvar

# class TemporalPatient:
#     def __init__(self, time_index, rating, outcome):
#         self.time_index = time_index
#         self.rating = rating
#         self.outcome = outcome


class TemporalPatientList:

    def __init__(self, pid, patient_data_list):
        self.pid = str(pid)
        self.data = patient_data_list

