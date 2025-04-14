    # def get_input_rows_dict(self):
    #     input_rows = []
    #     for datapoint in self.datapoints:
    #         input_rows.append(datapoint.input)

    #     return input_rows

    # def get_output_rows_dict(self):
    #     output_rows = []
    #     for datapoint in self.datapoints:
    #         output_rows.append(datapoint.output)
            
    #     return output_rows
    
    # def get_input_columns_dict(self):
    #     input_columns = {}
    #     for label in self.input_labels:
    #         input_column = []
    #         for datapoint in self.datapoints:
    #             input_column.append(datapoint.input[label])
    #         input_columns[label]= input_column

    #     return input_columns
         
    # def get_output_columns_dict(self):
    #     output_columns = {}
    #     for label in self.output_labels:
    #         output_column = []
    #         for datapoint in self.datapoints:
    #             output_column.append(datapoint.output[label])
    #         output_columns[label]= output_column

    #     return output_columns

    # def get_input_rows_list(self):
    #     input_rows = []
    #     for datapoint in self.datapoints:
    #         input_rows.append(list(datapoint.input.values()))

    #     return input_rows, self.input_labels

    # def get_output_rows_list(self):
    #     output_rows = []
    #     for datapoint in self.datapoints:
    #         output_rows.append(list(datapoint.output.values()))

    #     return output_rows, self.input_labels

    # def get_input_columns_list(self):
    #     input_columns = []
    #     for label in self.input_labels:
    #         input_column = []
    #         for datapoint in self.datapoints:
    #             input_column.append(datapoint.input[label])
    #         input_columns.append(input_column)

    #     return input_columns

    # def get_output_columns_list(self):
    #     output_columns = []
    #     for label in self.output_labels:
    #         output_column = []
    #         for datapoint in self.datapoints:
    #             output_column.append(datapoint.output[label])
    #         output_columns.append(output_column)

    #     return output_columns

    # def load_input_rows_dict(self, rows):
    #     dataset = Data_Set()

    #     for row in rows:
    #         datapoint = Data_Point(input = row)
    #         dataset.append_datapoint(datapoint)
        
    #     self.append_dataset(dataset)

    # def load_output_rows_dict(self, rows):
    #     dataset = Data_Set()

    #     for row in rows:
    #         datapoint = Data_Point(output = row)
    #         dataset.append_datapoint(datapoint)
        
    #     self.append_dataset(dataset)

    # def load_input_columns_dict(self, columns):
    #     dictionaries = [] 

    #     for jdx, label in enumerate(list(columns.keys())):
    #         for idx, label_row in enumerate(columns[label]):
    #             if jdx == 0:
    #                 dictionaries.append({label:label_row})
    #             else:
    #                 dictionaries[idx][label]=label_row

    #     self.load_input_rows_dict(dictionaries)

    # def load_output_columns_dict(self, columns):
    #     dictionaries = [] 

    #     for jdx, label in enumerate(list(columns.keys())):
    #         for idx, label_row in enumerate(columns[label]):
    #             if jdx == 0:
    #                 dictionaries.append({label:label_row})
    #             else:
    #                 dictionaries[idx][label]=label_row

    #     self.load_output_rows_dict(dictionaries)

    # def load_input_rows_list(self, rows, labels):
    #     dataset = Data_Set()

    #     for row in rows:
    #         dictionary = {}
    #         for idx, label in enumerate(labels):
    #             dictionary[label] = row[idx]
    #         datapoint = Data_Point(input = dictionary)
    #         dataset.append_datapoint(datapoint)
        
    #     self.append_dataset(dataset)

    # def load_output_rows_list(self, rows, labels):
    #     dataset = Data_Set()

    #     for row in rows:
    #         dictionary = {}
    #         for idx, label in enumerate(labels):
    #             dictionary[label] = row[idx]
    #         datapoint = Data_Point(output = dictionary)
    #         dataset.append_datapoint(datapoint)
        
    #     self.append_dataset(dataset)
    
    # def load_input_columns_list(self, columns, labels):
    #     dictionaries = [] 

    #     for jdx, label in enumerate(labels):
    #         for idx, label_row in enumerate(columns[jdx]):
    #             if jdx == 0:
    #                 dictionaries.append({label:label_row})
    #             else:
    #                 dictionaries[idx][label]=label_row
        
    #     self.load_input_rows_dict(dictionaries)

    # def load_output_columns_list(self, columns, labels):
    #     dictionaries = [] 

    #     for jdx, label in enumerate(labels):
    #         for idx, label_row in enumerate(columns[jdx]):
    #             if jdx == 0:
    #                 dictionaries.append({label:label_row})
    #             else:
    #                 dictionaries[idx][label]=label_row
        
    #     self.load_output_rows_dict(dictionaries)
