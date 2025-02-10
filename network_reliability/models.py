

# Create your models here.
from django.db import models

class YourModel(models.Model):
    field_name = models.CharField(max_length=100)
    # Ajoutez d'autres champs selon vos besoins

    def __str__(self):
        return self.field_name
