from django.db.models import Model, QuerySet, Manager


class SerializableQuerySet(QuerySet):
    pass


class SerializableModel(Model):
    objects = Manager.from_queryset(SerializableQuerySet)()

    class Meta:
        abstract = True
