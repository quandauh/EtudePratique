from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0015_item_cluster"),
    ]

    operations = [
        migrations.CreateModel(
            name="AutoEncoderTrainingRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Date d'entrainement")),
                ("nb_epochs", models.IntegerField(verbose_name="Nombre d'epochs")),
                ("hidden_size", models.IntegerField(verbose_name="Taille couche cachée")),
                ("output_file", models.CharField(max_length=255, verbose_name="Fichier de sortie")),
                ("final_loss", models.FloatField(blank=True, null=True, verbose_name="Loss finale")),
                ("final_false_positives", models.IntegerField(blank=True, null=True, verbose_name="False positives finaux")),
                ("final_false_negatives", models.IntegerField(blank=True, null=True, verbose_name="False negatives finaux")),
                ("total_duration_seconds", models.FloatField(blank=True, null=True, verbose_name="Durée totale")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
