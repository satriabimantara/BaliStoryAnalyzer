from django.urls import path, re_path
from narratives.views.home import index as home_index
from narratives.views.api_docs import index as api_docs_index
from narratives.views.license import index as license_index
from narratives.views.features import (
    framework_index,
    character_identification_view,
    character_classification_view,
    alias_clustering_view,
)

app_name = 'narratives'
urlpatterns = [
    path('', home_index, name='index'),
    path('features/framework/', framework_index, name='framework'),
    path('features/character-classification/',
         character_classification_view, name='chars_classify'),
    path('features/character-identification/',
         character_identification_view, name='chars_identify'),
    path('features/alias-clustering/',
         alias_clustering_view, name='alias_clustering'),
    path('api_docs/', api_docs_index, name='api_docs_index'),
    path('license/', license_index, name='license_index'),
]
