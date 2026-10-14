from django.db import migrations

class Migration(migrations.Migration):

    def create_person(apps, schema_editor):
        AppSettings = apps.get_model('app', 'AppSettings')
        AppSettings.objects.create(ae_hidden_size=25)

    dependencies = [
        ('app', '0013_appsettings'),
    ]

    operations = [
        migrations.RunPython(create_person)
    ]