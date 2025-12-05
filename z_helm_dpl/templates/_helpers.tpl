{{/*
Extended labels
*/}}
{{- define "postgres-exporter.labels" -}}
app: {{ include "postgres-exporter.name" . }}
chart: {{ include "postgres-exporter.chart" . }}
release: {{ .Release.Name }}
heritage: {{ .Release.Service }}
{{- end }}

{{/*
Chart name
*/}}
{{- define "postgres-exporter.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Chart full name
*/}}
{{- define "postgres-exporter.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart information
*/}}
{{- define "postgres-exporter.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Instance name
*/}}
{{- define "postgres-exporter.instance.name" -}}
{{- $fullname := include "postgres-exporter.fullname" . -}}
{{- printf "%s-%s" $fullname .instanceKey -}}
{{- end -}}

{{/*
Instance labels
*/}}
{{- define "postgres-exporter.instance.labels" -}}
{{ include "postgres-exporter.labels" . }}
instance: {{ .instanceKey }}
{{- end }}

{{/*
Data source name
*/}}
{{- define "postgres-exporter.datasource" -}}
{{- $dsn := printf "postgresql://%s:$(PASSWORD)@%s:%d/%s?sslmode=%s" 
  .instance.auth.username 
  .instance.database.host 
  .instance.database.port 
  .instance.database.name 
  .instance.database.sslmode -}}
{{- $dsn -}}
{{- end -}}