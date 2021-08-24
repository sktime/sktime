class ShapeletBase:

    def __init__(self, instance_index, start_pos, length, class_index, data):
        self.instance_index = instance_index
        self.start_pos = start_pos
        self.length = length
        self.quality = 0
        self.class_index = class_index
        self.data = data

    def __str__(self):
        return (
            "Series ID: {0}, start_pos: {1}, length: {2}, class: {3},  quality: {4} \n"
            " ".format(self.instance_index, self.start_pos, self.length, self.class_index, self.quality)
        )

    def get_class(self):
        return self.class_index

    def set_quality(self, q):
        self.quality = q

    __repr__ = __str__


class ShapeletDependent(ShapeletBase):

    def __init__(self, series_id, start_pos, length, quality, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.quality = quality
        self.data = data

    def __str__(self):
        return (
            "Series ID: {0}, start_pos: {1}, length: {2}, quality: {3}  \n"
            " ".format(self.series_id, self.start_pos, self.length, self.quality)
        )


class ShapeleIndependent(ShapeletBase):

    def __init__(self, instance_index, start_pos, length, class_index, dimension_id, data):
        super().__init__(instance_index, start_pos, length, class_index, data)
        self.dimension_id = dimension_id

    def __str__(self):
        return (
            "Series ID: {0}, start_pos: {1}, length: {2}, dimension ID: {3}, class: {4},  quality: {5}  \n"
            " ".format(self.instance_index, self.start_pos, self.length, self.dimension_id, self.class_index,
                       self.quality)
        )
