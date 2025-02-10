from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/data-entry/', views.data_entry, name='data_entry'),
    path('api/view-csv/', views.view_csv, name='view_csv'),
    path('api/build-model/', views.build_and_train_model_view, name='build_and_train_model'),
    path('api/report/', views.report, name='report'),
    path('static/images/<path:filename>', views.custom_static, name='custom_static'),
]
